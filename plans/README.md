# Reversed Reversi NCS：分阶段 Agent Plan 文件包

一次只向编程 Agent 提交一个当前可执行 Plan。每份 Plan 0–7 都包含固定设计基线、阶段任务、Agent 执行协议、指定状态报告文件名、交付物与验收条件。

| 文件 | 阶段 | 当前是否可执行 |
|---|---|---|
| `Plan_0_Task_0.md` | 项目现状检查 | 是，第一阶段 |
| `Plan_1_Task_1.md` | 博弈 AI 与固定特征评估器 | 是，待 Plan 0 通过后 |
| `Plan_2_Task_2.md` | 通用优化框架接口 | 是，待 Plan 1 通过后 |
| `Plan_3_Task_3.md` | 论文 NCS-C 实现 | 是，待 Plan 2 通过后 |
| `Plan_4_Task_4.md` | NCS-C 数学与合成函数测试 | 是，待 Plan 3 通过后 |
| `Plan_5_Task_5.md` | 对局评价适配器 | 是，待 Plan 4 通过后 |
| `Plan_6_Task_6.md` | NCS 与黑白棋端到端优化 | 是，待 Plan 5 通过后 |
| `Plan_7_Task_7.md` | 对照实验与最终提交 | 是，待 Plan 6 通过后 |
| `Plan_B_Future_Direction_Not_Executable.md` | 方案 B 方向备忘录 | **否，不要交给 Agent** |

项目根目录固定为 `Project1/`。每阶段开始都要递归读取 `Project1/docs/` 中全部可读文件，尤其是先前 `State n - Task n.md` 报告；完成后写入对应新报告，不得覆盖旧文件。方案 B 只有在用户与助手决定具体算法并另行生成可执行 Plan 后才进入状态编号 8。
