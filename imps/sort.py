import random
import time

def random_array(size, max):
    """ Returns an array with random integers

    size:  The number of items you want in the array.
    max:  The maximum negative and positive value in the array
    """

    arr = []
    for x in range(0, size):
        arr.append(random.randint(-max, max))
    return arr


def merge_sort(a):
    """ Merge sort implementation.  Takes an array and returns it sorted
    """

    if len(a) <= 1:
        return a
    else:
        midpoint = len(a)/2
        left_array = merge_sort(a[:midpoint])
        right_array = merge_sort(a[midpoint:])
        return merge(left_array, right_array)

def merge(left_array, right_array):
    """ Combines two subarrays into one sorted array
    """
    left_index, right_index, result = 0, 0, []
    while left_index < len(left_array) and right_index < len(right_array):
        if left_array[left_index] < right_array[right_index]:
            result.append(left_array[left_index])
            left_index += 1
        else:
            result.append(right_array[right_index])
            right_index += 1

    if left_index < len(left_array):
        result += left_array[left_index:]
    else:
        result += right_array[right_index:]

    return result

def insertion_sort(array):
    for j in range(1, len(array)):
        key = array[j]
        i = j - 1
        while i >= 0 and array[i] > key:
            array[i+1] = array[i]
            i = i - 1
        array[i+1] = key
    return array

def main():
    a = random_array(100000, 1000)

    t1= time.time()
    merge_sort_result = merge_sort(a)
    t2= time.time()
    print "merge sort took %s seconds" % (t2-t1)
    merge_sort_time = t2-t1

    t1= time.time()
    i_sort_result = insertion_sort(a)
    t2= time.time()
    print "insertion sort took %s seconds" % (t2-t1)
    i_sort_time = t2-t1

    print "merge sort ran %s times faster than insertion_sort" % (1.0*i_sort_time/merge_sort_time)

    #test to make sure sorts works
    #assert sorted(a) == merge_sort_result, "failed on %s, %s" % (a, derp)
    #assert sorted(a) == i_sort_result, "failed on %s, %s" % (a, derp)
    # print "tests pass"

if __name__ == "__main__":
    main()
