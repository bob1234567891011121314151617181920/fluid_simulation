pub(crate) fn pcg<A, M>(
    mut apply_a: A,
    mut apply_preconditioner: M,
    x: &mut [f32],
    b: &[f32],
    max_iterations: usize,
    tolerance: f32,
) -> bool
where
    A: FnMut(&[f32], &mut [f32]),
    M: FnMut(&[f32], &mut [f32]),
{
    debug_assert_eq!(x.len(), b.len());

    let n = b.len();

    let mut residual = vec![0.0; n];
    let mut z = vec![0.0; n];
    let mut direction = vec![0.0; n];
    let mut a_direction = vec![0.0; n];

    let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(&a, &b)| a * b).sum::<f32>();

    // residual = b - A*x
    apply_a(x, &mut a_direction);

    for i in 0..n {
        residual[i] = b[i] - a_direction[i];
    }

    let b_squared = dot(b, b);
    let tolerance_squared = tolerance * tolerance * b_squared.max(1.0);

    if dot(&residual, &residual) <= tolerance_squared {
        return true;
    }

    // z = M^-1 * residual
    apply_preconditioner(&residual, &mut z);

    // Initial search direction
    direction.copy_from_slice(&z);

    let mut residual_dot_z = dot(&residual, &z);

    if residual_dot_z <= 0.0 || !residual_dot_z.is_finite() {
        return false;
    }

    for _ in 0..max_iterations {
        // A * direction
        apply_a(&direction, &mut a_direction);

        let direction_dot_a_direction = dot(&direction, &a_direction);

        if direction_dot_a_direction <= 0.0 || !direction_dot_a_direction.is_finite() {
            return false;
        }

        let alpha = residual_dot_z / direction_dot_a_direction;

        for i in 0..n {
            x[i] += alpha * direction[i];
            residual[i] -= alpha * a_direction[i];
        }

        if dot(&residual, &residual) <= tolerance_squared {
            return true;
        }

        apply_preconditioner(&residual, &mut z);

        let new_residual_dot_z = dot(&residual, &z);

        if new_residual_dot_z <= 0.0 || !new_residual_dot_z.is_finite() {
            return false;
        }

        let beta = new_residual_dot_z / residual_dot_z;

        for i in 0..n {
            direction[i] = z[i] + beta * direction[i];
        }

        residual_dot_z = new_residual_dot_z;
    }

    false
}
