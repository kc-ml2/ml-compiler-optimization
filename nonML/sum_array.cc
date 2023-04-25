#include <iostream>
#include <cstdlib>
#include <ctime>


const int ARRAY_SIZE = 1000000;

void fill_random(int* array, int size) {
  for (int i = 0; i < size; i++) {
    array[i] = rand() % 100;
  }
}

int sum_array(int* array, int size) {
  int sum = 0;
  for (int i = 0; i < size; i++) {
    sum += array[i];
  }
  return sum;
}

int main() {
  int array[ARRAY_SIZE];
  fill_random(array, ARRAY_SIZE);

  // Perform a loop that calls sum_array multiple times
  // with random subsets of the array.
  std::srand(std::time(nullptr));
  int total_sum = 0;
  for (int i = 0; i < 1000; i++) {
    int start_index = std::rand() % ARRAY_SIZE;
    int subarray_size = std::rand() % (ARRAY_SIZE - start_index);
    int* subarray = array + start_index;
    int sum = sum_array(subarray, subarray_size);
    total_sum += sum;
  }

  std::cout << "Total sum: " << total_sum << std::endl;
  return 0;
}

