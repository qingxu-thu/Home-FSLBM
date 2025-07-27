4. check the bubble remove kernel
5. reuse
6. fix the u/rho usage (Not done)

7. 2d bubbles method change or check
8. delete the non-used memory/variable
9. change the boundary of 2D g (periodic?)


Record 2025/07/25
1. decide to rewrite some parts
    a. rewrite the non-kernel part
    b. rewrite the kernel part (surface part)
    c. rewrite the kernel part (bubble part)

Some issues:
1. I delete the remove parts since it seems to be never reference? (bubble remove and bubble remove last)
2. fix the lbm/ without vector (need to delete the things in initial)

Record 2025/07/26
1. decide to change the flag type? (MLLATTICENODE_SURFACE_FLAG -> unsigned char)
