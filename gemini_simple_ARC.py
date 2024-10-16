def max_subarray_sum(nums):
  max_so_far = nums[0]
  max_ending_here = nums[0]

  for num in nums[1:]:
    max_ending_here = max(num, max_ending_here + num)
    max_so_far = max(max_so_far, max_ending_here)

  return max_so_far

# Example usage
nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
nums=[-1,1,2]
result = max_subarray_sum(nums)
print(result)  # Output: 6