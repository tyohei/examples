#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


namespace py = pybind11;

py::array_t<double> add_array(py::array_t<double> arr1, py::array_t<double> arr2) {
  auto buf1 = arr1.request();
  auto buf2 = arr2.request();

  if (buf1.ndim != 1 && buf2.ndim != 1) {
    throw std::runtime_error("Number of dimension must be one.");
  }
  if (buf1.size != buf2.size) {
    throw std::runtime_error("Input shapes must match");
  }

  auto arr3 = py::array_t<double>(buf1.size);
  auto buf3 = arr3.request();
  
  double *ptr1 = (double *) buf1.ptr,
         *ptr2 = (double *) buf2.ptr,
         *ptr3 = (double *) buf3.ptr;

  for (size_t idx=0; idx<buf1.shape[0]; idx++) {
    ptr3[idx] = ptr1[idx] + ptr2[idx];
  }

  return arr3;
}


PYBIND11_MODULE(test, m) {
    m.def("add_array", &add_array, "Add two NumPy arrays");
}
