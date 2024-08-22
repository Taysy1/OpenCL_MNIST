__kernel void FullConnect(__global float* matrix,
    __global float* vector,
    __global float* bias,
    __global float* result,
    int N) 
{ 
    int row = get_global_id(0); // 每个工作项对应矩阵的一行

    __private float dot_product = 0.0;

    for (int col = 0; col < N; ++col) {
        dot_product += matrix[row * N + col] * vector[col];
        if (col == N - 1) {
            // 最后加上偏置bias
            dot_product += bias[row];
        }
    }

    result[row] = dot_product;
}
