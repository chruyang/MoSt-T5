# AutoDL 文件存储迁移交接（2026-07-30）

## 已完成的快照

- 快照目录：`/root/autodl-fs/migrations/most-t5-20260730T211141+0800`
- 迁移完成：`2026-07-30T21:46:13+08:00`
- 校验完成：`2026-07-30T22:12:33+08:00`
- 文件存储占用约 86 GB，剩余约 115 GB。

## 已保全内容

- 完整活动工作树：`MoSt-T5/`（约 41 GB，含 `.git`、未提交代码、checkpoint、词表、日志和数据）。
- 数据/基线：`3D-MoIT/`（约 17 GB）、`3D-MolT5/`（约 3 GB）及四个 `e3fp-*` 下游目录。
- Git 备用包：`most-t5-origin-main.bundle`。
- Phase-1 恢复候选：`recovery_candidates/phase1/pubchemqc_final.lmdb` 及锁文件。
- 历史候选：`recovery_candidates/legacy_lmdb/` 与 `sparse_artifacts/`。
- 环境重建记录：`environment/3dmolt5-conda-explicit.txt`、`environment/3dmolt5-environment.yml`。

没有迁移 `.autodl/` 平台状态，也没有无差别复制整个 `.Trash-0/`。

## 稀疏 LMDB 的处理

`/root/autodl-fs` 是 FUSE 文件存储，不保留稀疏文件洞。以下文件不能直接展开到该目录：

- `3D-MoIT/.../pubchemqc_e3fp.lmdb`：逻辑 1 TB、实际约 8 KiB。
- 回收站中的 `pubchemqc_e3fp.lmdb` 与 `pubchemqc_final_FAT.lmdb`：逻辑各 1 TB。

它们已用 GNU tar 的 sparse 格式保存：

- `sparse_artifacts/3D-MoIT_pubchemqc_e3fp.lmdb.tar`
- `sparse_artifacts/recovery_pubchemqc_e3fp_and_final_FAT.tar`

在新机器的支持稀疏文件的数据盘中恢复，例如：

```bash
tar --sparse -xpf /root/autodl-fs/migrations/most-t5-20260730T211141+0800/sparse_artifacts/3D-MoIT_pubchemqc_e3fp.lmdb.tar -C /root/autodl-tmp/3D-MoIT
```

不要在 `/root/autodl-fs` 内解包这些归档。归档成员清单、源端稀疏元数据和 SHA-256 位于 `verification/`。

## 校验结论

- 13 个常规对象的 `rsync --checksum --dry-run` 结果均为空，表示内容一致。
- `MoSt-T5`：源/目标均为 1,064 个常规文件、3 个符号链接；链接目标一致。
- `3D-MoIT`：源端 104 个常规文件、目标 103 个；唯一排除项正是已归档的稀疏 LMDB。
- AutoDL FUSE 将三个符号链接的权限映射为 `0644`（源端 `0777`）；链接权限对符号链接解析无影响，且链接目标已核对一致。

## 迁移后仍需处理的 P0 问题

`run_train.sh` 引用的 Phase-1 活动路径 `3D-MoIT/3d-mol-dataset/pubchemqc/pubchemqc_final.lmdb` 在原活动目录中不存在。相同文件名的回收站副本已隔离在 `recovery_candidates/phase1/`，但尚未做血缘/内容审计，因此本迁移**没有**把它自动恢复到活动训练路径。

## 实用提示

目标快照中的三个顶层历史符号链接使用旧的绝对路径；若新机器不沿用 `/root/autodl-tmp/MoSt-T5`，应在确认指向的 checkpoint runs 已存在后重建这些链接。根目录里的 `most-t5-sparse-probe-20260730` 是迁移前的 64 MiB 稀疏能力测试文件，不属于研究资产。
