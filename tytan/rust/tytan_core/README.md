# tytan_core

TYTAN の Rust 拡張モジュールです。Python から呼ばれる高速化層として、
差分計算・バッチ差分・結果集計・SA step を Rust 側に寄せています。

## この Rust 化のコンセプト

Rust にした目的は、単に Python を置き換えることではなく、**FFI 往復を減らしつつ
ホットパスを Rust に集約する**ことです。

- **差分計算**: `delta_energy` / `batch_delta` で、1回ごとの差分評価を高速化
- **step 処理**: `sa_step_single_flip` / `sa_step_multi_flip` で、複数 step を Rust 側にまとめる
- **返却経路の最適化**: Rust から返す配列を Python 側で組み直しすぎない
- **対称 Q の最適化**: 対称行列では簡略式を使い、両側参照を避ける
- **contiguous 前提**: 連続メモリの配列を前提に、コピーを最小化する

この方針により、最新の計測では Phase 3 の Rust step が pure Python より高速になりました。

## 構成

- `src/lib.rs` : PyO3 の公開関数と Python との橋渡し
- `src/delta.rs` : 差分エネルギー計算
- `src/anneal.rs` : SA step 実装
- `src/pool.rs` : Adaptive pool 管理
- `src/adaptive.rs` : AdaptiveBulkSASampler の Rust コア
- `src/reduce.rs` : 結果集計
- `src/types.rs` : 共通バリデーションと補助関数

## 公開 API

- `delta_energy` : 単一ビット反転の差分計算
- `batch_delta` : 複数 state の差分をまとめて計算
- `aggregate_results` : state/energy を結果形式へ集約
- `sa_step_single_flip` : 1 step 分の SA を Rust 側で実行
- `sa_step_multi_flip` : 複数 step を Rust 側でまとめて実行
- `adaptive_bulk_sa` : Adaptive bulk SA の Rust コア

## Python 側との接続

Python からは `tytan/_rust_backend.py` を経由して呼び出します。

主な制御変数:

- `TYTAN_RUST` : `off` / `auto` / `on`
- `TYTAN_RUST_MIN_WORK` : batch delta を使う最小ワーク量
- `TYTAN_RUST_STEP_MIN_WORK` : step 全体を Rust に寄せる最小ワーク量

Rust 側は contiguous な `float64` / `int64` 配列を前提にしています。
入力が不連続な場合は Python 側で正規化します。

## ビルド

### Rust crate のビルド

```bash
cargo build --release
```

### Python 拡張の開発ビルド

```bash
VIRTUAL_ENV=/path/to/.venv PATH=/path/to/.venv/bin:$PATH \
  maturin develop --release --manifest-path tytan/rust/tytan_core/Cargo.toml
```

インストール後、Python は `tytan._tytan_rust` を優先して読み込みます。

## テスト

```bash
cargo test
pytest tests/rust/test_rust_backend_fallback.py
pytest tests/rust/test_rust_backend_parity.py
pytest tests/rust/test_adaptive_bulk_rust_batch_path.py
```

## ベンチマーク

Phase 3 専用ベンチ:

```bash
PYTHONPATH=. python tools/bench_phase3.py
```

最新の計測では、Rust step は pure Python より高速で、batch delta よりも速い結果でした。

## 実装メモ

- `delta.rs` は対称 Q の fast path を持つ
- `anneal.rs` は multi-step 実行に対応する
- `lib.rs` は Python への返却を直接配列化する
- `tytan/_rust_backend.py` は Rust 利用可否と入力正規化を担当する
