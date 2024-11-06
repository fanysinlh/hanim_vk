use hanim_vk::backend::window_main;
use winit::dpi::{Size, PhysicalSize};

fn main() {
    let data = window_main::AppData {
        window_size: Size::new(PhysicalSize::<i32>{width: 800, height: 600}),
    };
    let mut window = window_main::HanimApp::init(data);
    window.run();
}
