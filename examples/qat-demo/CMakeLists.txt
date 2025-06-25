add_executable(qat-demo qat-demo.cpp)
target_link_libraries(qat-demo PRIVATE ggml)
target_include_directories(qat-demo PRIVATE ../../include)