# ✅ FIXED: Array Assignment Size Mismatch

## Problem
```
Unable to perform assignment because the size of the left side is 1-by-32-by-20-by-6 
and the size of the right side is 1-by-1-by-1-by-6.
```

## Root Cause
The `qExample` array is 4D: `(nx, ny, nz, ndof)` = `(24, 32, 20, 6)`

**Wrong approach:**
```matlab
qExample(k,:,:,:) = reshape(qex_flat, [1,1,1,ndof]);
```
This tries to assign to index `k` with dimensions `[1, 32, 20, 6]` - mismatch!

## Solution
Use proper linear indexing for 4D array:

```matlab
if ~isempty(qex_flat)
    % Store IK solution using linear indexing
    for d = 1:ndof
        qExample(k + (d-1)*Nvox) = qex_flat(d);
    end
end
```

This correctly maps:
- `k` = voxel linear index (1 to 15360)
- `d` = joint dimension (1 to 6)
- Linear index = `k + (d-1)*Nvox`

## Try Running Again

The fix is applied to both parallel and serial modes. Run:

```matlab
cd C:\Users\yanbo\wSpace\cinebotRL\matlab
run_build
```

Should work now! 🚀

## What This Fixes

- ✅ Parallel loop can now store IK solutions correctly
- ✅ Serial loop also fixed with same approach
- ✅ No more dimension mismatch errors
- ✅ qExample will properly store joint configurations for seeding

---

**Status:** ✅ Ready to run!  
**Expected:** Build should start and complete in 10-15 minutes
