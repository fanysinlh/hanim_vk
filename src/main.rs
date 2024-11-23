use hanim_vk::backend::window_main;
use winit::dpi::{Size, PhysicalSize};

fn main() {
    let mut hobject_data = vec![];
    hobject_data.push(window_main::HobjectData {
        position: [-0.5, 0.0, 1.8],
        color: [1.0, 0.0, 0.0, 1.0],
    });
    hobject_data.push(window_main::HobjectData {
        position: [0.5, 0.0, 0.0],
        color: [0.0, 1.0, 0.0, 1.0],
    });
    hobject_data.push(window_main::HobjectData {
        position: [0.0, 0.5, 0.0],
        color: [0.0, 0.0, 1.0, 1.0],
    });

    let data = window_main::AppData {
        window_size: Size::new(PhysicalSize::<i32>{width: 800, height: 600}),
        hobject_data,
    };
    let mut window = window_main::HanimApp::init(data);
    window.run();
}
