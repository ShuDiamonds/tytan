# tytan_core (Phase 1)

Rust extension module for TYTAN Phase 1 acceleration.

## Build (local)

```bash
cargo build --release
```

## Build Python extension (development)

```bash
maturin develop --manifest-path tytan/rust/tytan_core/Cargo.toml
```

When installed, Python can import the module as `tytan._tytan_rust` (preferred) or `_tytan_rust` depending on build setup.
