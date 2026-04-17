# Adaptive SA テスト

このディレクトリには、TYTAN に追加した Adaptive SA スイートの単体テストが入っています。新規機能と補助モジュールが意図したとおりに動くかを着実に確認するための内容です。

## 確認していること
1. **Reference SA ベースライン (`test_reference_sa.py`)**
   - CPU 上で単一軌道で実行し、返却構造が `dict/energy/count` の形で安定しているか確認します。
   - `return_stats=True` を指定したときに統計情報（改善回数、ステップ数、ショット数）を含んで返すことも検証します。
2. **DeltaEvaluator (`test_delta_evaluator.py`)**
   - 全更新時のエネルギーを再計算した結果と、差分インクリメント（1 ビットフリップ）の出力が一致するかをチェックします。
   - `local_field()` で局所場を計算できることも確認します。
3. **SolutionPool (`test_solution_pool.py`)**
   - 現在/ベスト/多様解を保持し、重複を抑えながら結果整形（`to_results()`）が想定どおりになるかをテストします。
   - 見やすい順序（エネルギー昇順）で並び、`diverse_k` に応じた多様解が末尾に追加されるかを確認します。
4. **StrategyManager (`test_strategy_manager.py`)**
   - ε-greedy で高重み戦略を優先する挙動と、ε=1.0 のときに探索的に選ばれることをそれぞれ確認します。
   - 報酬（reward）を記録して戦略の重みが増加することも検証します。
5. **ClampManager (`test_clamp_manager.py`)**
   - soft/hard clamp 候補のスコア更新と、`lock()` → `apply()` で状態変化が反映されるかを確認します。
6. **AdaptiveBulkSASampler (`test_adaptive_bulk_sa.py`)**
   - 複数戦略・複数軌道を束ねて実行し、結果が `list` で返されることと、統計情報（最良エネルギー、戦略重み、ログ）を含むことを検証します。

## 実行方法
```bash
uv run pytest tests/adaptive_sa
```

`uv` を使うことで、プロジェクトに必要な依存関係を同期したうえでテストが実行できます。`uv run` で Python 仮想環境内から `pytest` を起動するため、他の環境を汚さずに済みます。

## 環境準備
- `uv sync --all-groups` により依存モジュールを整備してください。`vcrpy`, `numpy`, `pytest` などを `pyproject.toml` で管理しています。
- macOS や Linux 上で `python` 3.11 系が動いていれば動作します。

## 想定する使い方
- 各モジュールを編集したらこのテスト群を回して、基本的な挙動に崩れがないか確認します。
- 既存の `sampler.py` に統合したり、さらに GPU API を書く前段階としてまず CPU 上での挙動を確かめるのに使ってください。
