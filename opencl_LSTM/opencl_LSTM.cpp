#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <cassert>
#include <string>
#include <vector>
#include <CL/cl.h>
#include"params.h"
#include <random>

using namespace std;

//错误C4996表示clCreateCommandQueue函数已被弃用。在OpenCL 2.0及更高版本中，应使用clCreateCommandQueueWithProperties函数替代
//但是我比较懒，直接禁用此警告
#pragma warning(disable: 4996)

//根据列表的id选择平台
//平台信息：
//[0] Intel(R) OpenCL HD Graphics
//[1] NVIDIA CUDA
//[2] Intel(R) OpenCL (CPU)
//[3] Intel(R) FPGA Emulation Platform for OpenCL(TM)
//[4] Intel(R) FPGA SDK for OpenCL(TM)


// 检查返回值错误
#define CHECK_ERRORS(ERR) \
	if(ERR != CL_SUCCESS){ \
		cerr << "OpenCL error code: " << ERR << "  file: " << __FILE__  << "  line: " << __LINE__ << ".\nExiting..." << endl; \
		exit(1); \
	}


int select_platform_id;
int main(int argc, const char** argv)
{
    cl_int err = 0;
    cl_platform_id* platforms_list = NULL;

    // 3.1 获取 platform name的长度
    size_t platform_name_length = 0;
    // 3.2 根据name的长度分配
    char* platform_name_value = new char[platform_name_length];

    // 1. 获取平台数量
    cl_uint platforms_num;
    err = clGetPlatformIDs(0, nullptr, &platforms_num);
    CHECK_ERRORS(err);


    // 2. 获取所有平台的信息
    if (platforms_num > 0) {
        cl_platform_id* platforms = new cl_platform_id[platforms_num];
        platforms_list = platforms;
        err = clGetPlatformIDs(platforms_num, platforms, nullptr);
        CHECK_ERRORS(err);

        cout << "OpenCL Platform Info:" << endl;
        // 3. 遍历获取各个平台的信息
        for (int i = 0; i < platforms_num; i++)
        {

            err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, nullptr, &platform_name_length);
            CHECK_ERRORS(err);

            err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, platform_name_length, platform_name_value, nullptr);
            CHECK_ERRORS(err);

            cout << "    [" << i << "]: " << platform_name_value << endl;
        }

        cout << "请需要输入使用的平台id" << endl;
        cin >> select_platform_id;


        if (select_platform_id <= platforms_num - 1) {
            err = clGetPlatformInfo(platforms_list[select_platform_id], CL_PLATFORM_NAME, 0, nullptr, &platform_name_length);
            CHECK_ERRORS(err);
            err = clGetPlatformInfo(platforms_list[select_platform_id], CL_PLATFORM_NAME, platform_name_length, platform_name_value, nullptr);
            CHECK_ERRORS(err);
            cout << "\n★ Find The Platform（" << select_platform_id << "）: " << platforms[select_platform_id] << endl;
        }
        else {
            cout << "\n☆ Can't Find The Platform! " << endl;
            exit(-1);
            // 遍历各个平台都没找到想要的平台，则退出
        }
    }


    // 4. 创建平台上下文
    cl_platform_id platform = platforms_list[select_platform_id];
    delete[] platforms_list;

    cl_context_properties ctx_prop[3] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platform,
        0
    };

    cl_context ctx = clCreateContextFromType(ctx_prop, CL_DEVICE_TYPE_ALL, NULL, NULL, &err);
    CHECK_ERRORS(err);


    // 5. 获取平台设备数目
    size_t devices_size = 0;
    err = clGetContextInfo(ctx, CL_CONTEXT_DEVICES, 0, NULL, &devices_size);
    CHECK_ERRORS(err);


    // 6. 申请平台设备数组, 获取所有设备信息
    cl_device_id* devices = new cl_device_id[devices_size];
    err = clGetContextInfo(ctx, CL_CONTEXT_DEVICES, devices_size, devices, NULL);
    CHECK_ERRORS(err);

    // 7. C++ 读取 .cl 程序源码
    string CodeFileName = "cl_code.cl";
    std::ifstream kernelFile(CodeFileName, std::ios::in);
    if (!kernelFile.is_open())
    {
        std::cerr << "Failed to open file for reading: " << CodeFileName << std::endl;
        return NULL;
    }
    std::ostringstream oss;
    oss << kernelFile.rdbuf();

    std::string srcCode = oss.str();
    const char* cl_StrCode = srcCode.c_str();


    // 8. 初始化程序
    cl_program  program = clCreateProgramWithSource(ctx, 1, (const char**)&cl_StrCode, NULL, &err);
    CHECK_ERRORS(err);


    // 9. 编译程序
    err = clBuildProgram(program, 1, devices, NULL, NULL, NULL);
    // After clBuildProgram
    size_t length;
    char buffer[2048];
    clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
    printf("--- Build log ---\n%s\n", buffer);
    CHECK_ERRORS(err);


    // 10. 创建内核 
    cl_kernel kernel = clCreateKernel(program, "lstm", &err);


    // 11. 创建命令队列
    cl_command_queue cmd_q = clCreateCommandQueue(ctx, devices[0], CL_QUEUE_PROFILING_ENABLE, &err);
    //cl_command_queue cmd_q2 = clCreateCommandQueue(ctx, devices[0], CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_ERRORS(err);


    // 12. 初始化数据

    // Example parameters
    //const int inputSize = 1;
    //const int hiddenSize = 2;
    //const int timeSteps = 1;
    const int batchSize = 1;

    // Create random input data

    std::vector<float> input(batchSize* timeSteps* inputSize, 1.0f);
    //float input[timeSteps * inputSize] = {1.0f};
    std::vector<float> output(batchSize* hiddenSize);


    //initializeWeights(inputWeights, hiddenSize * (inputSize + hiddenSize));
    //initializeWeights(forgetWeights, hiddenSize * (inputSize + hiddenSize));
    //initializeWeights(outputWeights, hiddenSize * (inputSize + hiddenSize));
    //initializeWeights(candidateWeights, hiddenSize * (inputSize + hiddenSize));
    //initializeWeights(inputBias, hiddenSize);
    //initializeWeights(forgetBias, hiddenSize);
    //initializeWeights(outputBias, hiddenSize);
    //initializeWeights(candidateBias, hiddenSize);

    // 13. 创建buffer
    cl_mem inputBuffer = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * timeSteps * inputSize, input.data(), &err);
    CHECK_ERRORS(err);
    cl_mem outputBuffer = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, sizeof(float) * output.size(), nullptr, &err);
    CHECK_ERRORS(err);
    cl_mem inputWeightsBuffer = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * hiddenSize * (inputSize + hiddenSize), inputWeights, &err);
    CHECK_ERRORS(err);
    cl_mem forgetWeightsBuffer = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * hiddenSize * (inputSize + hiddenSize), forgetWeights, &err);
    CHECK_ERRORS(err);
    cl_mem outputWeightsBuffer = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * hiddenSize * (inputSize + hiddenSize), outputWeights, &err);
    CHECK_ERRORS(err);
    cl_mem candidateWeightsBuffer = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * hiddenSize * (inputSize + hiddenSize), candidateWeights, &err);
    CHECK_ERRORS(err);
    cl_mem inputBiasBuffer = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * hiddenSize, inputBias, &err);
    CHECK_ERRORS(err);
    cl_mem forgetBiasBuffer = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * hiddenSize, forgetBias, &err);
    CHECK_ERRORS(err);
    cl_mem outputBiasBuffer = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * hiddenSize, outputBias, &err);
    CHECK_ERRORS(err);
    cl_mem candidateBiasBuffer = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * hiddenSize, candidateBias, &err);
    CHECK_ERRORS(err);

    // 14. 将数据写入buffer队列
    //err = clEnqueueWriteBuffer(cmd_q, cmM, CL_TRUE, 0, mem_size_M, M, 0, NULL, NULL);           CHECK_ERRORS(err);
    //err = clEnqueueWriteBuffer(cmd_q, cmV, CL_TRUE, 0, mem_size_V, V, 0, NULL, NULL);           CHECK_ERRORS(err);

    //err = clEnqueueWriteBuffer(cmd_q, cm_bias, CL_TRUE, 0, mem_size_bias, bias, 0, NULL, NULL); CHECK_ERRORS(err);
    //err = clEnqueueWriteBuffer(cmd_q, cmW, CL_TRUE, 0, mem_size_W, W, 0, NULL, NULL);           CHECK_ERRORS(err);


    // 15. 设置内核参数
 // Set kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);            CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);           CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &inputWeightsBuffer);     CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &inputBiasBuffer);        CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 4, sizeof(cl_mem), &outputWeightsBuffer);    CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 5, sizeof(cl_mem), &outputBiasBuffer);       CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 6, sizeof(cl_mem), &forgetWeightsBuffer);    CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 7, sizeof(cl_mem), &forgetBiasBuffer);       CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 8, sizeof(cl_mem), &candidateWeightsBuffer); CHECK_ERRORS(err);
    err = clSetKernelArg(kernel, 9, sizeof(cl_mem), &candidateBiasBuffer);    CHECK_ERRORS(err);


    // 16. 将内核放入命令队列
    cl_event kernel_event;               // OpenCL event
    size_t szGlobalWorkSize;        // Total # of work items in the 1D range
    size_t szLocalWorkSize;         // # of work items in the 1D work group    
    size_t szParmDataBytes;         // Byte size of context information
    size_t szKernelLength;          // Byte size of kernel code
    cl_ulong start_time_1, end_time_1, start_time_2, end_time_2;

    //szLocalWorkSize = 256;
    szGlobalWorkSize = batchSize;
    //执行内核
    err = clEnqueueNDRangeKernel(cmd_q, kernel, 1, NULL, &szGlobalWorkSize, NULL, 0, NULL, &kernel_event);
    CHECK_ERRORS(err);

    // 17. 等待队列命令执行完毕
    err = clFinish(cmd_q); CHECK_ERRORS(err);


    // 18. 取出计算后的数据
    err = clEnqueueReadBuffer(cmd_q, outputBuffer, CL_TRUE, 0, sizeof(float) * output.size(), output.data(), 0, nullptr, nullptr); CHECK_ERRORS(err);
    // 20. Calculate and print the execution time
    cl_ulong start_time, end_time, kernelExeTime_ns=0;
    err = clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL); CHECK_ERRORS(err);
    err = clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL); CHECK_ERRORS(err);
    kernelExeTime_ns += end_time - start_time;
    cout<<"Kernel Execution time in nano_seconds for LSTM Layer:"<< kernelExeTime_ns <<endl;


    for (int i = 0; i < batchSize; ++i) {
        std::cout << "Output for Batch " << i << ": ";
        for (int j = 0; j < hiddenSize; ++j) {
            std::cout << output[i * hiddenSize + j] << std::fixed << setprecision(10) << " ";
        }
        std::cout << std::endl;
    }


    // Clean up
    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseMemObject(inputWeightsBuffer);
    clReleaseMemObject(forgetWeightsBuffer);
    clReleaseMemObject(outputWeightsBuffer);
    clReleaseMemObject(candidateWeightsBuffer);
    clReleaseMemObject(inputBiasBuffer);
    clReleaseMemObject(forgetBiasBuffer);
    clReleaseMemObject(outputBiasBuffer);
    clReleaseMemObject(candidateBiasBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(cmd_q);
    clReleaseContext(ctx);


    return 0;

}

//void initializeWeights(float* weights, int size) {
//    std::random_device rd;
//    std::mt19937 gen(rd());
//    std::uniform_real_distribution<> dis(-1.0, 1.0);
//
//    for (int i = 0; i < size; ++i) {
//        weights[i] = dis(gen);
//    }
//}








