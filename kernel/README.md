# Stock Lenovo kernel binaries (TB710FU)

From the stock TB710FU_CN ZUXOS_1.5.04.470 (260625) firmware.
The kernel itself is built from source (android14-6.1 GKI, TB520FU kernel repo).

- `dtb/`, `dtbo.img`: stock device trees (PRC board).
- `modules/vendor_boot/`, `modules/vendor_dlkm/`: stock vendor kernel modules and load lists.

Not managed by extract-files: a full extraction clears this directory,
so restore it with `git checkout -- kernel` afterwards.
