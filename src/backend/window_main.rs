use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferExecFuture, CommandBufferUsage, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents, SubpassEndInfo};
use vulkano::image::view::ImageView;
use vulkano::sync::future::{FenceSignalFuture, JoinFuture};
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::shader::ShaderModule;
use vulkano::sync::{self, GpuFuture};
use vulkano::{swapchain, Validated, VulkanLibrary};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::device::{Queue, QueueFlags};
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::swapchain::{PresentFuture, Surface, SwapchainAcquireFuture, SwapchainPresentInfo};
use vulkano::device::DeviceExtensions;
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::image::{Image, ImageUsage};
use vulkano::swapchain::{Swapchain, SwapchainCreateInfo};
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::VulkanError;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::run_return::EventLoopExtRunReturn;
use winit::window::{Window, WindowBuilder};
use std::sync::Arc;
use super::super::shaders::{fragment_shaders::fs, vertex_shaders::vs};

#[derive(BufferContents, Vertex, Clone)]
#[repr(C)]
pub struct HobjectData {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32G32B32A32_SFLOAT)]
    pub color: [f32; 4],
}

pub struct AppData {
    pub window_size: winit::dpi::Size,
    pub hobject_data: Vec<HobjectData>,
}

pub struct HanimApp {
    event_loop: EventLoop<()>,
    hanim_window: HanimWindow,
}

struct HanimWindow {
    window: Arc<Window>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    swapchain: Arc<Swapchain>,
    render_pass: Arc<RenderPass>,
    vertex_buffer: Subbuffer<[HobjectData]>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    viewport: Viewport,
    command_buffer_allocator: StandardCommandBufferAllocator,
    command_buffers: Vec<Arc<PrimaryAutoCommandBuffer>>,
    fences: Vec<Option<Arc<FenceSignalFuture<PresentFuture<
        CommandBufferExecFuture<
            JoinFuture<Box<dyn GpuFuture>, SwapchainAcquireFuture>
        >>>>>>,
    previous_fence_i: usize,
    window_resized: bool,
    recreate_swapchain: bool,
}

impl HanimApp {
    pub fn init(data: AppData) -> Self {
        let (event_loop, instance) = Self::event_and_instance_init();

        let (window, surface) = 
            Self::window_and_service_init(&data, &event_loop, &instance);

        let (physical_device, device, queue) = 
            Self::device_and_queue_init(&instance, &surface);

        let (swapchain, images) = 
            Self::swachain_and_images_init(&window, &device, &surface, &physical_device);

        let (render_pass, vertex_buffer, vs, fs, viewport) = 
            Self::vertex_buffer_and_shaders_and_viewport_init(&data, &window, &device, &swapchain);

        let frame_buffers = Self::get_framebuffers(&images, &render_pass);
    
        let pipeline = Self::get_pipeline(
            device.clone(),
            vs.clone(),
            fs.clone(),
            render_pass.clone(),
            viewport.clone(),
        );
    
        let command_buffer_allocator = StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        );
    
        let command_buffers = Self::get_command_buffers(
            &command_buffer_allocator,
            &queue,
            &pipeline,
            &frame_buffers,
            &vertex_buffer,
        );
    
        let window_resized = false;
        let recreate_swapchain = false;
    
        // 使用栅栏来实现“飞行中的帧”，在GPU渲染前一帧的同时，CPU处理后一帧
        let frames_in_flight = images.len();
        let fences: Vec<Option<Arc<FenceSignalFuture<_>>>> = vec![None; frames_in_flight];
        let previous_fence_i = 0;

        Self {
            event_loop,
            hanim_window: HanimWindow {
                window,
                device,
                queue,
                swapchain,
                render_pass,
                vertex_buffer,
                vs,
                fs,
                viewport,
                command_buffer_allocator,
                command_buffers,
                fences,
                previous_fence_i,
                window_resized,
                recreate_swapchain,
            },
        }
    }

    pub fn run(&mut self) {
        self.hanim_window.show(&mut self.event_loop);
    }

    fn event_and_instance_init() -> (EventLoop<()>, Arc<Instance>) {
        let event_loop = EventLoop::new();
        let library = VulkanLibrary::new().expect("no local Vulkan library/DLL");
        let required_extensions = Surface::required_extensions(&event_loop);
        let instance = Instance::new(
            library, 
            InstanceCreateInfo {
                enabled_extensions: required_extensions,
                ..Default::default()
            })
            .expect("failed to create instance");
        (event_loop, instance)
    }

    fn window_and_service_init(data: &AppData, event_loop: &EventLoop<()>, instance: &Arc<Instance>)
        -> (Arc<Window>, Arc<Surface>) {
        let window = Arc::new(WindowBuilder::new()
            .with_title("Hanim")
            .with_inner_size(data.window_size)
            .build(&event_loop)
            .unwrap());
        let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();

        (window, surface)
    }

    fn device_and_queue_init(instance: &Arc<Instance>, surface: &Arc<Surface>)
        -> (Arc<PhysicalDevice>, Arc<Device>, Arc<Queue>) {
        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };
    
        let (physical_device, _) = Self::select_physical_device(
            &instance, &surface, &device_extensions);
    
        // 获取支持图形操作的队列族
        let queue_family_index = physical_device
            .queue_family_properties()
            .iter()
            .enumerate()
            .position(|(_queue_family_index, queue_family_properties)| {
                queue_family_properties.queue_flags.contains(QueueFlags::GRAPHICS)
            })
            .expect("couldn't find a graphical queue family") as u32;
        
        let (device, mut queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_extensions: device_extensions,
                ..Default::default()
            },
        )
        .expect("failed to create device");
    
        let queue = queues.next().unwrap();

        (physical_device, device, queue)
    }

    fn swachain_and_images_init(
        window: &Arc<Window>,
        device: &Arc<Device>,
        surface: &Arc<Surface>,
        physical_device: &Arc<PhysicalDevice>,
    ) -> (Arc<Swapchain>, Vec<Arc<Image>>) {
        // 查询一下表面的能力
        let caps = physical_device
            .surface_capabilities(&surface, Default::default())
            .expect("failed to get surface capabilities");
    
        // 获取图像的尺寸、透明度处理、以及图像格式（如RGB）。
        let dimensions = window.inner_size();
        let composite_alpha = caps.supported_composite_alpha.into_iter().next().unwrap();
        let image_format = physical_device
            .surface_formats(&surface, Default::default())
            .unwrap()[0]
            .0;
    
        let (swapchain, images) = Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: caps.min_image_count + 1, // 使用的交换链数量
                // 为了更灵活的图像队列处理，最好将min_image_count设置为至少比最小值多一个。
                image_format,
                image_extent: dimensions.into(),
                image_usage: ImageUsage::COLOR_ATTACHMENT,  // 图像的用途
                composite_alpha,
                ..Default::default()
            },
        )
        .unwrap();

        (swapchain, images)
    }

    fn vertex_buffer_and_shaders_and_viewport_init(
        data: &AppData,
        window: &Arc<Window>,
        device: &Arc<Device>,
        swapchain: &Arc<Swapchain>,
    ) -> (Arc<RenderPass>, Subbuffer<[HobjectData]>, Arc<ShaderModule>, Arc<ShaderModule>, Viewport) {
        let render_pass = Self::get_render_pass(device.clone(), &swapchain);
    
        // 使用默认内存分配器，等下分配给缓冲区
        let memory_allocator = 
            Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let vertex_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            data.hobject_data.clone(),
        )
        .unwrap();

        let vs = vs::load(device.clone()).expect("failed to create shader module");
        let fs = fs::load(device.clone()).expect("failed to create shader module");
    
        let viewport = Viewport {
            // 视口从窗口左下角开始，偏移量为0
            offset: [0.0, 0.0],
            // 指定视口的宽度和高度
            extent: window.inner_size().into(),
            // 视口的深度范围（由近到远），近平面为0，原平面为1
            depth_range: 0.0..=1.0,
        };

        (render_pass, vertex_buffer, vs, fs, viewport)
    }

    fn select_physical_device(
        instance: &Arc<Instance>,
        surface: &Arc<Surface>,
        device_extensions: &DeviceExtensions,
    ) -> (Arc<PhysicalDevice>, u32) {
        instance
            .enumerate_physical_devices()
            .expect("could not enumerate devices")
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            // filter_map：映射后为None就去掉，映射后为Some(value)，就保留value
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    // position：根据条件。过滤出第一个满足的，然后返回它的索引
                    .position(|(i, q)| {
                        q.queue_flags.contains(QueueFlags::GRAPHICS)
                            && p.surface_support(i as u32, &surface).unwrap_or(false)
                    })
                    .map(|q| (p, q as u32))
            })
            // min_by_key：把每个元素映射为对应的key，然后比较key，返回key最小的元素
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                _ => 4,
            })
            .expect("no device available")
    }    

    fn get_render_pass(device: Arc<Device>, swapchain: &Arc<Swapchain>) -> Arc<RenderPass> {
        vulkano::single_pass_renderpass!(
            device,
            attachments: {
                color: {
                    // 设置格式与交换链相同。
                    format: swapchain.image_format(),
                    samples: 1,
                    // 我们希望在进入通道时，将通道用单一颜色填充。
                    load_op: Clear,
                    // 我们希望把图像存储进文件
                    store_op: Store,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {},
            },
        )
        .unwrap()
    }
    
    fn get_framebuffers(
        images: &[Arc<Image>],
        render_pass: &Arc<RenderPass>,
    ) -> Vec<Arc<Framebuffer>> {
        images
            .iter()
            .map(|image| {
                let view = ImageView::new_default(image.clone()).unwrap();
                Framebuffer::new(
                    render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![view],
                        ..Default::default()
                    },
                )
                .unwrap()
            })
            .collect::<Vec<_>>()
    }
    
    fn get_pipeline(
        device: Arc<Device>,
        vs: Arc<ShaderModule>,
        fs: Arc<ShaderModule>,
        render_pass: Arc<RenderPass>,
        viewport: Viewport,
    ) -> Arc<GraphicsPipeline> {
        // 获取入口点（进入着色器执行的函数）
        let vs = vs.entry_point("main").unwrap();
        let fs = fs.entry_point("main").unwrap();
    
        // 配置顶点输入状态，
        // per_vertex()会根据MyVertex的定义创建顶点输入的描述信息，
        // 然后传给definition，根据顶点着色器的输入接口自动匹配数据格式。
        let vertex_input_state = HobjectData::per_vertex()
            .definition(&vs.info().input_interface)
            .unwrap();
    
        let stages = [
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs),
        ];
    
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )
        .unwrap();
    
        // 创建唯一的一个子通道，作为图形渲染管线的目标
        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();
    
        GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                // 设置输入装配状态，定义顶点的连接方式（例如，使用三角形列表、线条或点）。
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState {
                    viewports: [viewport].into_iter().collect(),
                    ..Default::default()
                }),
                // 设置光栅化阶段状态，控制像素生成（默认设置为直接显示三角形）。
                rasterization_state: Some(RasterizationState::default()),
                // 多重采样状态，控制抗锯齿设置（默认为无多重采样）。
                multisample_state: Some(MultisampleState::default()),
                // 颜色混合状态，用于定义如何将输出颜色与帧缓冲区中已有颜色混合。
                // ColorBlendState::with_attachment_states 是一个静态方法，用于创建颜色混合状态。
                // 它允许我们为渲染的每个附件（即目标帧缓冲区中的颜色缓冲）定义颜色混合规则。
                // 这里调用了 with_attachment_states 方法，该方法接受两个参数：附件数量和颜色混合状态。
                // subpass.num_color_attachments() 获取当前渲染子通道（subpass）中颜色附件的数量。
                // 如果子通道有多个颜色附件，那么就可能需要为每个附件指定一个混合状态。
                // 在这种情况下，这个数字会告诉 Vulkan 需要创建多少个颜色混合状态。
                // 默认状态（ColorBlendAttachmentState::default()）通常表示不执行任何特殊混合，而是简单地覆盖已有颜色。
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.num_color_attachments(),
                    ColorBlendAttachmentState::default(),
                )),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .unwrap()
    }
    
    fn get_command_buffers(
        command_buffer_allocator: &StandardCommandBufferAllocator,
        queue: &Arc<Queue>,
        pipeline: &Arc<GraphicsPipeline>,
        framebuffers: &Vec<Arc<Framebuffer>>,
        vertex_buffer: &Subbuffer<[HobjectData]>,
    ) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
        framebuffers
            .iter()
            .map(|framebuffer| {
                let mut builder = AutoCommandBufferBuilder::primary(
                    command_buffer_allocator,
                    queue.queue_family_index(),
                    // 不要忘记写入正确的缓冲区使用。
                    CommandBufferUsage::MultipleSubmit,
                )
                .unwrap();
    
                builder
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            clear_values: vec![Some([0.1, 0.1, 0.1, 1.0].into())],
                            ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                        },
                        SubpassBeginInfo {
                            // Inline表示当前命令缓冲区包含了所有渲染命令
                            contents: SubpassContents::Inline,
                            ..Default::default()
                        },
                    )
                    .unwrap()
                    .bind_pipeline_graphics(pipeline.clone())
                    .unwrap()
                    // 把顶点缓冲区绑定到管线的第0个输入槽位
                    .bind_vertex_buffers(0, vertex_buffer.clone())
                    .unwrap()
                    // 顶点和实例的数量、顶点和实例的起始位置（偏移量）
                    .draw(vertex_buffer.len() as u32, 1, 0, 0)
                    .unwrap()
                    .end_render_pass(SubpassEndInfo::default())
                    .unwrap();
    
                builder.build().unwrap()
            })
            .collect()
    }    
}


impl HanimWindow {
    pub fn show(&mut self, event_loop: &mut EventLoop<()>) {
        event_loop.run_return(move |event, _, control_flow| match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                self.window_resized = true;
            }
            // MainEventsCleared在所有输入事件处理完毕并即将开始重绘时发出。这使我们可以编写每帧的渲染功能。
            Event::MainEventsCleared => {
                // 处理窗口大小改变、需要重建交换链的情况
                if self.window_resized || self.recreate_swapchain {
                    self.recreate_swapchain = false;
                
                    let new_dimensions = self.window.inner_size();
                
                    if new_dimensions.width > 0 && new_dimensions.height > 0 {
                        let (new_swapchain, new_images) = self.swapchain
                            .recreate(SwapchainCreateInfo {
                                image_extent: new_dimensions.into(),
                                ..self.swapchain.create_info()
                            })
                            .expect("failed to recreate swapchain: {e}");
                        self.swapchain = new_swapchain;
                        let new_framebuffers = HanimApp::get_framebuffers(&new_images, &self.render_pass);
                
                        if self.window_resized {
                            self.window_resized = false;
                    
                            self.viewport.extent = new_dimensions.into();
                            let new_pipeline = HanimApp::get_pipeline(
                                self.device.clone(),
                                self.vs.clone(),
                                self.fs.clone(),
                                self.render_pass.clone(),
                                self.viewport.clone(),
                            );
                            self.command_buffers = HanimApp::get_command_buffers(
                                &self.command_buffer_allocator,
                                &self.queue,
                                &new_pipeline,
                                &new_framebuffers,
                                &self.vertex_buffer,
                            );
                        }
                    }
                }
                // 从交换链中获取图像
                // image_i是图像索引，suboptimal如果为true，说明图像不够理想，可能无法显示；
                // acquire_future表示GPU获得此图像的时刻（实际上应该是多线程里的future概念）
                let (image_i, suboptimal, acquire_future) =
                // acquire_next_image的第二个参数，是一个超时值。如果超过那个值还是没获取到图像，就报错
                match swapchain::acquire_next_image(self.swapchain.clone(), None)
                    .map_err(Validated::unwrap)
                {
                    Ok(r) => r,
                    Err(VulkanError::OutOfDate) => {
                        self.recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("failed to acquire next image: {e}"),
                };
                // 如果图像不理想，重建交换链
                if suboptimal {
                    self.recreate_swapchain = true;
                }
    
                // 等待与该图像相关的 fence 完成。通常这会是最早的 fence，可能已经完成。
                if let Some(image_fence) = &self.fences[image_i as usize] {
                    image_fence.wait(None).unwrap();
                }
    
                // 使用前一帧的future进行关联，只需在future不存在时同步
                let previous_future = match self.fences[self.previous_fence_i as usize].clone() {
                    // 创建一个 `NowFuture`。
                    None => {
                        let mut now = sync::now(self.device.clone());
                        // cleanup_finished：函数会释放所有未使用的资源
                        now.cleanup_finished();
                        // 调用boxed把future存储到堆上，因为其大小不确定
                        now.boxed()
                    }
                    // 使用现有的 `FenceSignalFuture`。
                    Some(fence) => fence.boxed(),
                };
    
                let future = previous_future
                    .join(acquire_future)
                    // then_execute将特定的命令从命令缓冲区，提交到队列中执行
                    .then_execute(self.queue.clone(), self.command_buffers[image_i as usize].clone())
                    .unwrap()
                    // then_swapchain_present将渲染好的图像呈现在屏幕上
                    .then_swapchain_present(
                        self.queue.clone(),
                        SwapchainPresentInfo::swapchain_image_index(self.swapchain.clone(), image_i),
                    )
                    // 利用信号量，确保GPU确实执行了该操作，并在操作完成后向CPU发出信号。
                    .then_signal_fence_and_flush();
    
                // 把旧的fence用于错误处理
                self.fences[image_i as usize] = match future.map_err(Validated::unwrap) {
                    Ok(value) => Some(Arc::new(value)),
                    Err(VulkanError::OutOfDate) => {
                        self.recreate_swapchain = true;
                        None
                    }
                    Err(e) => {
                        println!("failed to flush future: {e}");
                        None
                    }
                };
                self.previous_fence_i = image_i as usize;
            }
            _ => (),
        });
    }
}
