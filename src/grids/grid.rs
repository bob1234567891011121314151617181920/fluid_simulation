use glam::UVec3;

pub struct Grid3D<T> {
    dimensions: UVec3,
    default_value: T,
    data: Vec<T>,
}

impl<T: Clone> Clone for Grid3D<T> {
    fn clone(&self) -> Self {
        Self {
            dimensions: self.dimensions,
            default_value: self.default_value.clone(),
            data: self.data.clone(),
        }
    }
}

impl<T: Clone> Grid3D<T> {
    pub fn new(x: u32, y: u32, z: u32, default_value: T) -> Self {
        Self {
            dimensions: UVec3::new(x, y, z),
            default_value: default_value.clone(),
            data: vec![default_value; x as usize * y as usize * z as usize],
        }
    }

    fn index(&self, x: u32, y: u32, z: u32) -> usize {
        let x = x as usize;
        let y = y as usize;
        let z = z as usize;
        let dimensions_y = self.dimensions.y as usize;
        let dimensions_z = self.dimensions.z as usize;

        x * dimensions_y * dimensions_z + y * dimensions_z + z
    }

    pub fn dimensions(&self) -> UVec3 {
        self.dimensions
    }

    pub fn get(&self, x: u32, y: u32, z: u32) -> T {
        let index = self.index(x, y, z);
        self.data[index].clone()
    }

    pub fn get_mut(&mut self, x: u32, y: u32, z: u32) -> &mut T {
        let index = self.index(x, y, z);
        &mut self.data[index]
    }

    pub fn set(&mut self, x: u32, y: u32, z: u32, value: T) {
        let index = self.index(x, y, z);
        self.data[index] = value;
    }

    pub fn clear(&mut self) {
        self.data.fill(self.default_value.clone());
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum CellType {
    Air,
    Fluid,
    Solid,
}
