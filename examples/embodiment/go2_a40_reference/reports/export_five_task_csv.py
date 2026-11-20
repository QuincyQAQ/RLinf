# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Export the retained final A40 measurements as one Excel-readable CSV."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


def main():
    main_result = read("five_task_final_20261120.json")
    a1_file = "completed_a1_confirmatory_14032589.json"
    a1_repeat = read(a1_file)
    b1_file = "completed_b1_confirmatory_14036295.json"
    b1_repeat = read(b1_file)
    tune_file = "completed_a1_policy_tune_14034565.json"
    tune = read(tune_file)
    assert all(
        e["data"]["status"] == "verified" for e in main_result["results"].values()
    )
    assert all(r["status"] == "verified" for r in [a1_repeat, b1_repeat, tune])
    rows = []

    def add(task, phase, policy, updates, seeds, values, source, notes=""):
        count = sum(v["num_episodes"] for v in values)
        successes = sum(round(v["num_episodes"] * v["success_rate"]) for v in values)

        def mean(key, factor=1):
            if not all(key in v for v in values):
                return ""
            return factor * sum(v[key] * v["num_episodes"] for v in values) / count

        passage = (
            sum(v["physical_passage_successes"] for v in values)
            if all("physical_passage_successes" in v for v in values)
            else ""
        )
        rows.append(
            [
                task,
                phase,
                policy,
                0,
                seeds,
                updates,
                successes,
                count,
                100 * successes / count,
                mean("mean_lin_vel_error"),
                mean("mean_ang_vel_error"),
                mean("failure_rate", 100),
                mean("mean_success_step_ratio", 100),
                mean("mean_alive_time_s"),
                passage,
                mean("physical_obstacle_passage_rate", 100),
                mean("joint_tracking_and_passage_success_rate", 100),
                notes,
                source,
            ]
        )

    names = {
        "baseline": "原 PPO 基线",
        "rsl_control": "继续普通 PPO",
        "rlinf_huber": "继续 RLinf-Huber",
        "rlinf_conservative_huber": "继续 RLinf-Huber",
        "rlinf_conservative_mse": "继续 RLinf-MSE",
    }
    tasks = {
        "A1": ("A1 前进", 3000),
        "A2": ("A2 全向移动", 1500),
        "B1": ("B1 粗糙地形", 3000),
        "B2": ("B2 楼梯", 2200),
        "C2": ("C2 沟壑", 3000),
    }
    for task, (label, budget) in tasks.items():
        result = main_result["results"][task]["data"]
        lanes = (
            ["baseline", "rsl_control", "rlinf_huber"]
            if task != "B2"
            else [
                "baseline",
                "rsl_control",
                "rlinf_conservative_huber",
                "rlinf_conservative_mse",
            ]
        )
        for lane in lanes:
            note = "两个评测 seed 分别评测后按回合数合并；训练 seed 只有 0"
            if task == "B1":
                note += "；修复后源策略，三组统一动作历史限幅 +/-100"
            if task in ["B2", "C2"]:
                note += "；跟踪成功不等于实际障碍通过"
            add(
                label,
                "主实验" if lane != "rlinf_conservative_mse" else "补充损失函数分支",
                names[lane],
                budget + (1000 if lane != "baseline" else 0),
                "23456、34567",
                list(result["results"][lane].values()),
                "five_task_final_20261120.json",
                note,
            )
        if task == "A1":
            for lane in ["baseline", "rsl_control", "rlinf_huber"]:
                add(
                    label,
                    "固定策略补测",
                    names[lane],
                    3000 if lane == "baseline" else 4000,
                    "45678",
                    [a1_repeat["new_seed_results"][lane]],
                    a1_file,
                    "新评测 seed；固定原 checkpoint；未混入主实验统计",
                )
            for lane, name, updates in [
                ("untuned", "调参前 RLinf", 4000),
                ("low_entropy", "RLinf：熵系数 0.001", 4500),
                ("zero_entropy", "RLinf：熵系数 0", 4500),
            ]:
                add(
                    label,
                    "调参验证／选型，非最终评测",
                    name,
                    updates,
                    "56789",
                    [tune["all_validation_results"][lane]],
                    tune_file,
                    "验证集结果；非最终成绩"
                    + ("；选中候选" if lane == "zero_entropy" else ""),
                )
            for lane, name, updates in [
                ("ordinary_ppo_4000", "普通 PPO，未追加训练", 4000),
                ("ordinary_ppo_4500", "普通 PPO 再续训 500 轮", 4500),
                ("untuned_rlinf_4000", "调参前 RLinf，未追加训练", 4000),
                ("selected_rlinf", "调参后 RLinf：熵系数 0", 4500),
            ]:
                add(
                    label,
                    "调参实验最终评测",
                    name,
                    updates,
                    "67890、78901",
                    list(tune["holdout_results"][lane].values()),
                    tune_file,
                    "两个预留评测 seed；56789 验证选型后冻结策略；不与主实验混合",
                )
        if task == "B1":
            for lane in ["baseline", "rsl_control", "rlinf_huber"]:
                add(
                    label,
                    "固定策略补测",
                    names[lane],
                    3000 if lane == "baseline" else 4000,
                    "45678",
                    [b1_repeat["new_seed_results"][lane]],
                    b1_file,
                    "三组同一新 seed；各 2048 回合；未混入主实验；统一限幅 +/-100",
                )

    header = [
        "任务",
        "实验阶段",
        "策略／分组",
        "训练 seed",
        "评测 seed",
        "总训练轮数",
        "跟踪成功回合数",
        "评测回合数",
        "跟踪成功率（%）",
        "平均线速度误差（m/s）",
        "平均偏航角速度误差（rad/s）",
        "失败终止率（%）",
        "平均达标步比例（%）",
        "平均存活时间（s）",
        "实际障碍通过回合数",
        "实际障碍通过率（%）",
        "同回合跟踪且通过率（%）",
        "备注",
        "结果来源 JSON（相对本表目录）",
    ]
    assert len(rows) == 29
    assert all(len(row) == len(header) for row in rows)
    destination = ROOT / "A40_five_tasks_complete_20261121.csv"
    with destination.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    with destination.open(encoding="utf-8-sig", newline="") as f:
        verified = list(csv.DictReader(f))
    assert len(verified) == 29 and {r["任务"].split()[0] for r in verified} == set(
        tasks
    )
    assert all(int(r["跟踪成功回合数"]) <= int(r["评测回合数"]) for r in verified)
    assert destination.read_bytes().startswith(b"\xef\xbb\xbf")
    print(destination)
    print(
        f"Verified {len(verified)} records, {len(header)} columns; UTF-8 BOM; missing metrics blank."
    )


if __name__ == "__main__":
    main()
