cmake_minimum_required(VERSION 3.15...4.4)
project(cpp_test LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)

find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_binning NB_STATIC src/finufft_pk/_binning.cpp NOMINSIZE)
target_compile_options(_binning PRIVATE -O3 -march=native -ffast-math) 
#-fopenmp

install(TARGETS _binning LIBRARY DESTINATION finufft_pk)