add_executable(main main.cpp)
target_link_libraries(main PRIVATE lib)

configure_file(simulation_test_input.txt simulation_test_input.txt COPYONLY)