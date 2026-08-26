struct Globals {
    view_projection: mat4x4<f32>,
    sphere_radius: f32,
};

@group(0) @binding(0)
var<uniform> globals: Globals;

struct VertexInput {
    @location(0) unit_position: vec3<f32>,
    @location(1) sphere_position: vec3<f32>,
};

struct VertexOutput {
    @builtin(position)  clip_position: vec4<f32>,
};

@vertex
fn vertex_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    let world_position = input.unit_position * globals.sphere_radius + input.sphere_position;
    output.clip_position = globals.view_projection * vec4<f32>(world_position, 1.0);
    return output;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let water_color = vec3<f32>(0.0, 0.0, 1.0);
    return vec4<f32>(water_color, 1.0);
}
