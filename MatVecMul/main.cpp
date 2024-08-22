#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

using namespace std;

//����C4996��ʾclCreateCommandQueue�����ѱ����á���OpenCL 2.0�����߰汾�У�Ӧʹ��clCreateCommandQueueWithProperties�������
//�����ұȽ�����ֱ�ӽ��ô˾���
#pragma warning(disable : 4996)

int err;
#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
        std::cerr << "OpenCL error " << err << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

std::string readKernelSource(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open kernel source file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}


cl_platform_id selectPlatform() {
    cl_platform_id* platforms_list = NULL;
    // 3.1 ��ȡ platform name�ĳ���
    size_t platform_name_length = 0;
    // 3.2 ����name�ĳ��ȷ���
    char* platform_name_value = new char[platform_name_length];

    // 1. ��ȡƽ̨����
    cl_uint platforms_num;
    err = clGetPlatformIDs(0, nullptr, &platforms_num);
    CHECK_ERROR(err);

    // 2. ��ȡ����ƽ̨����Ϣ
    if (platforms_num > 0) {
        cl_platform_id* platforms = new cl_platform_id[platforms_num];
        platforms_list = platforms;
        err = clGetPlatformIDs(platforms_num, platforms, nullptr);
        CHECK_ERROR(err);

        cout << "OpenCL Platform Info:" << endl;
        // 3. ������ȡ����ƽ̨����Ϣ
        for (int i = 0; i < platforms_num; i++)
        {
            err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, nullptr, &platform_name_length);
            CHECK_ERROR(err);

            err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, platform_name_length, platform_name_value, nullptr);
            CHECK_ERROR(err);

            cout << "    [" << i << "]: " << platform_name_value << endl;

        }
        int select_platform_id;
        cout << "����Ҫ����ʹ�õ�ƽ̨id" << endl;
        cin >> select_platform_id;
        cl_platform_id selected_platform = platforms[select_platform_id];

        if (select_platform_id <= platforms_num - 1) {
            err = clGetPlatformInfo(platforms_list[select_platform_id], CL_PLATFORM_NAME, 0, nullptr, &platform_name_length);
            CHECK_ERROR(err);
            err = clGetPlatformInfo(platforms_list[select_platform_id], CL_PLATFORM_NAME, platform_name_length, platform_name_value, nullptr);
            CHECK_ERROR(err);
            cout << "\n�� Find The Platform��" << select_platform_id << "��: " << platforms[select_platform_id] << endl;
        }
        else {
            cout << "\n�� Can't Find The Platform! " << endl;
            exit(-1);// ��������ƽ̨��û�ҵ���Ҫ��ƽ̨�����˳�
        }
        return selected_platform;
    }
}


void mat_vec_mul_opencl(
    const std::vector<float>& matrix, 
    const std::vector<float>& vector,
    const std::vector<float>& bias,
    std::vector<float>& result, 
    int M, int N) {
    cl_int err;

    // 1. ��ȡƽ̨���豸

    cl_platform_id platform = selectPlatform();
    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL); CHECK_ERROR(err);

    // 2. ���������ĺ��������
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err); CHECK_ERROR(err);
    cl_queue_properties properties[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, properties, &err);
    CHECK_ERROR(err);


    // 3. �����ͱ������
    std::string kernel_source = readKernelSource("mat_vec_mul.cl"); CHECK_ERROR(err);
    const char* kernel_source_cstr = kernel_source.c_str();
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_source_cstr, NULL, &err); CHECK_ERROR(err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        std::vector<char> build_log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), NULL);
        std::cerr << "Error in kernel:\n" << build_log.data() << std::endl;
        exit(EXIT_FAILURE);
    }

    // 4. �����ں�
    cl_kernel kernel = clCreateKernel(program, "FullConnect", &err); CHECK_ERROR(err);

    // 5. ����������
    cl_mem matrix_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, M * N * sizeof(float), (void*)matrix.data(), &err); CHECK_ERROR(err);
    cl_mem vector_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * sizeof(float), (void*)vector.data(), &err); CHECK_ERROR(err);
    cl_mem bias_buf   = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, M * sizeof(float), (void*)bias.data(), &err); CHECK_ERROR(err);
    cl_mem result_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, M * sizeof(float), NULL, &err); CHECK_ERROR(err);


    // 6. �����ں˲���
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &matrix_buf); CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &vector_buf); CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bias_buf);   CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &result_buf); CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 4, sizeof(int), &N);             CHECK_ERROR(err);

    // 7. ����NDRange
    size_t global_work_size = M; //global_work_size ָ����Ҫִ���ں˵Ĺ�������������NDrange���work item�������ڸ������У�ÿ���������Ӧ�����һ�У���� global_work_size ����Ϊ������������� M��
    size_t local_work_size = 64; // ѡ����ʵĹ������С��ͨ����64��128�ı���
    if (M % local_work_size != 0) {
        global_work_size = (M / local_work_size + 1) * local_work_size;
    }

    // 8. ִ���ں�
    cl_event kernel_event;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &kernel_event); CHECK_ERROR(err);

    // 9. ��ȡ���
    err = clEnqueueReadBuffer(queue, result_buf, CL_TRUE, 0, M * sizeof(float), result.data(), 0, NULL, NULL); CHECK_ERROR(err);


    // 20. Calculate and print the execution time
    cl_ulong start_time, end_time;
    err = clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL); CHECK_ERROR(err);
    err = clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL); CHECK_ERROR(err);
    cout << endl << " Kernel_sq Execution time in nano_seconds: " << end_time - start_time << endl;

    // 10. ����
    clReleaseMemObject(matrix_buf);
    clReleaseMemObject(vector_buf);
    clReleaseMemObject(result_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);

    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

int main() {
    int M = 5;
    int N = 4;

    std::vector<float> matrix = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,
        10, 20, 20, 20
    };

    std::vector<float> vector = { 1, 1, 1, 0};
    std::vector<float> bias =  { 0, -1, -1, -1, -1 };
    std::vector<float> result(M, 0);

    mat_vec_mul_opencl(matrix, vector, bias, result, M, N);

    std::cout << "Result: ";
    for (int i = 0; i < M; ++i) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
