class Solution:
    def reverse(self, x: int) -> int:
        
        if x < 0:
            sign = -1
        else:
            sign = 1
            
        x = abs(x)

        rev = 0
        while x != 0:
            digit = x % 10
            x = x // 10
            if rev > 214748364:
                return 0
            rev = rev * 10 + digit
        rev = rev * sign

        return rev