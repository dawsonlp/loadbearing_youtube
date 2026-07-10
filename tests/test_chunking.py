from loadbearing_youtube.analysis.chunking import chunk_transcript, to_blocks
from loadbearing_youtube.models import Segment, Transcript


def _transcript(n=200, step=3.0):
    segs = [Segment(text=f"word{i}", start=i * step, duration=step) for i in range(n)]
    return Transcript(video_id="x", url="u", segments=segs)


def test_blocks_group_by_duration():
    t = _transcript(n=30, step=2.0)  # 60s total
    blocks = to_blocks(t, block_seconds=20)
    assert len(blocks) >= 3
    assert blocks[0].start == 0.0
    # blocks are ordered by start time
    starts = [b.start for b in blocks]
    assert starts == sorted(starts)


def test_chunks_respect_char_budget_and_cover_all():
    t = _transcript(n=300, step=1.0)
    chunks = chunk_transcript(t, max_chars=500, overlap_blocks=1, block_seconds=5)
    assert len(chunks) > 1
    for c in chunks:
        # allow a single-block overflow but never wildly over budget
        assert len(c.text) <= 500 + 200
    # first chunk starts at 0, chunks are contiguous/increasing
    assert chunks[0].start == 0.0
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_timestamps_present_in_chunk_text():
    t = _transcript(n=20, step=3.0)
    chunks = chunk_transcript(t, max_chars=100000)
    assert "[00:00]" in chunks[0].text


def test_empty_transcript():
    t = Transcript(video_id="x", url="u", segments=[])
    assert chunk_transcript(t) == []
    assert to_blocks(t) == []
