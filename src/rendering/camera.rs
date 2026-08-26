use glam::{
    Mat4, UVec3, Vec3,
    camera::rh::{proj::directx::perspective, view::look_at_mat4},
};

pub fn create_static_camera(dimensions: UVec3, screen_width: u32, screen_height: u32) -> Mat4 {
    let max_dimension = dimensions.max_element() as f32;

    let domain_size = dimensions.as_vec3() / max_dimension;
    let domain_center = domain_size * 0.5;
    let eye = domain_center + Vec3::new(1.5, 0.0, 0.2);
    let view = look_at_mat4(eye, domain_center, Vec3::Y);

    let aspect = screen_width as f32 / screen_height as f32;
    let projection = perspective(45_f32.to_radians(), aspect, 0.01, 10.0);

    projection * view
}
