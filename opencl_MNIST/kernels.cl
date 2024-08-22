#define hiddenSize 8
#define inputSize 9
#define timeSteps 10

inline float RELU(float x) {
    return (x > 0) ? x : 0;
}

inline float SIGMOID(float x) {
    return 1.0f / (1.0f + exp(-x));
}

//#define TANH(x) (exp(x) - exp(-(x))) / (exp(x) + exp(-(x)))
inline float TANH(float x) {
    return ((exp(x) - exp(-x))/(exp(x) + exp(-x)));
}

__kernel void lstm(__global float* input,            __global float* output,
                   __global float* inputWeights,     __global float* inputBias,
                   __global float* outputWeights,    __global float* outputBias,
                   __global float* forgetWeights,    __global float* forgetBias,
                   __global float* candidateWeights, __global float* candidateBias) {
    // ��ȡ��ǰ�̵߳�ȫ��ID
    int gid = get_global_id(0);


    // ��ʼ��LSTM��Ԫ״̬�����
    __private float vec_input[inputSize + hiddenSize];
    __private float cellState[hiddenSize] = { 0 }; // ����hiddenSize <= 1024
    __private float outputState[hiddenSize] = { 0 };
    __private float inputGate[hiddenSize];
    __private float forgetGate[hiddenSize];
    __private float outputGate[hiddenSize];
    __private float candidateCellState[hiddenSize];

    // ����ÿ��ʱ�䲽
    for (int t = 0; t < timeSteps; t++) {

        // ����LSTM cell0���������� ����ht-1��xtƴ�ӳ�[xt, ht-1]
        for (int j = 0; j < inputSize; j++)		vec_input[j] = input[t * inputSize + j];
        for (int j = 0; j < hiddenSize; j++)	vec_input[inputSize + j] = outputState[j];

         //��ʼ��LSTM�������š������š�����źͺ�ѡϸ��״̬
        for (int i = 0; i < hiddenSize; i++) {
            inputGate[i] = 0.0f;
            forgetGate[i] = 0.0f;
            outputGate[i] = 0.0f;
            candidateCellState[i] = 0.0f;
        }
        //printf("\n");
        // ���������š������š�����źͺ�ѡϸ��״̬��Ȩ�غ�ƫ��
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < inputSize+hiddenSize; j++) {
                //printf("inputWeights: %f, * vec_input: %f\n", inputWeights[i * (inputSize + hiddenSize) + j], vec_input[j]);
                inputGate[i]          += inputWeights[ i * (inputSize + hiddenSize)+j]  *  vec_input[j];
                forgetGate[i]         += forgetWeights[i * (inputSize + hiddenSize) + j] * vec_input[j];
                outputGate[i]         += outputWeights[i * (inputSize + hiddenSize) + j] * vec_input[j];
                candidateCellState[i] += candidateWeights[i * (inputSize + hiddenSize) + j] * vec_input[j];
            }

        }

        for (int i = 0; i < hiddenSize; i++) {

            inputGate[i] +=inputBias[i];
            forgetGate[i]+=forgetBias[i];
            outputGate[i]+=outputBias[i];
            candidateCellState[i]+=candidateBias[i];
        }


        for (int i = 0; i < hiddenSize; i++) {

            inputGate[i] = SIGMOID(inputGate[i]);
            forgetGate[i] = SIGMOID(forgetGate[i]);
            outputGate[i] = SIGMOID(outputGate[i]);
            candidateCellState[i] = TANH(candidateCellState[i]);

            //printf("inputGate: %d, value: %.10f\n", i, inputGate[i]);
            //printf("forgetGate: %d, value: %f\n", i, forgetGate[i]);
            //printf("outputGate: %d, value: %.10f\n", i, outputGate[i]);
            //printf("candidateCellState: %d, value: %.10f\n", i, candidateCellState[i]);
        }


        // ���������š������š�����źͺ�ѡϸ��״̬����LSTM��Ԫ״̬�����״̬
        for (int i = 0; i < hiddenSize; i++) {
            __private float temp1 = cellState[i];
            cellState[i] = inputGate[i] * candidateCellState[i] + forgetGate[i] * temp1;
            outputState[i] = TANH(cellState[i]) * outputGate[i];
            //printf("outputGate: %d, value: %.10f\n", i, outputGate[i]);
            //printf("TANH(cellState[i]): %d, value: %.10f\n", i, TANH(cellState[i]));
        }
    }
    // �����һ�����״̬д�����������
    for (int i = 0; i < hiddenSize; i++) {
        output[i] = outputState[i];
        //printf("output: %d, value: %.10f\n", i, output[i]);
    }
}


__kernel void FullConnect(__global float* matrix, 
                          __global float* vector, 
                          __global float* bias, 
                          __global float* result,  
                          int N){ // NΪ������
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
