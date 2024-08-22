__kernel void FullConnect(__global float* matrix,
    __global float* vector,
    __global float* bias,
    __global float* result,
    int N) 
{ 
    int row = get_global_id(0); // ÿ���������Ӧ�����һ��

    __private float dot_product = 0.0;

    for (int col = 0; col < N; ++col) {
        dot_product += matrix[row * N + col] * vector[col];
        if (col == N - 1) {
            // ������ƫ��bias
            dot_product += bias[row];
        }
    }

    result[row] = dot_product;
}
