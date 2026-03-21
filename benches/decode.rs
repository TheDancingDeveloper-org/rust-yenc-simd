use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use yenc::{decode_yenc, encode_article};

fn bench_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode");

    for size in [1_024, 64_000, 768_000] {
        let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let (encoded, _) = encode_article(&data, "bench.bin", 1, 2, 0, size as u64 * 10);

        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::new("yenc", size),
            &encoded,
            |b, encoded| {
                b.iter(|| {
                    let result = decode_yenc(encoded).unwrap();
                    std::hint::black_box(result);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_decode);
criterion_main!(benches);
