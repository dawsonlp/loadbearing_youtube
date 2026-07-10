import pytest

from loadbearing.sources.base import TranscriptError
from loadbearing.sources.youtube import YouTubeSource, extract_video_id


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://www.youtube.com/watch?v=8mY9wx_iMSU", "8mY9wx_iMSU"),
        ("https://youtu.be/8mY9wx_iMSU", "8mY9wx_iMSU"),
        ("https://www.youtube.com/watch?v=8mY9wx_iMSU&feature=share", "8mY9wx_iMSU"),
        ("https://youtube.com/embed/8mY9wx_iMSU", "8mY9wx_iMSU"),
        ("https://www.youtube.com/shorts/8mY9wx_iMSU", "8mY9wx_iMSU"),
        ("https://www.youtube.com/live/8mY9wx_iMSU", "8mY9wx_iMSU"),
        ("8mY9wx_iMSU", "8mY9wx_iMSU"),
    ],
)
def test_extract_video_id(url, expected):
    assert extract_video_id(url) == expected


def test_extract_video_id_invalid():
    with pytest.raises(TranscriptError):
        extract_video_id("https://example.com/not-a-video")


def test_source_matches():
    src = YouTubeSource()
    assert src.matches("https://youtu.be/abc")
    assert src.matches("8mY9wx_iMSU")
    assert not src.matches("https://vimeo.com/123")
