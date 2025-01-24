add_executable(pr pr.cu)
target_link_libraries(pr ${TORCH_LIBRARIES})
set_target_properties(pr PROPERTIES 
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)

add_executable(gcn gcn.cu)
target_link_libraries(gcn ${TORCH_LIBRARIES})
set_target_properties(gcn PROPERTIES 
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)