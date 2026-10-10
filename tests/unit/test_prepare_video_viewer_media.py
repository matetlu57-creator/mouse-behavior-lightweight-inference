from pathlib import Path
from types import SimpleNamespace

from scripts import prepare_video_viewer_media as media


def test_convert_resolves_relative_output_root(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    output_root = Path("outputs")
    source = (tmp_path / output_root / "rendered" / "clip.avi").resolve()
    source.parent.mkdir(parents=True)
    source.touch()
    monkeypatch.setattr(media.imageio_ffmpeg, "get_ffmpeg_exe", lambda: "ffmpeg")

    def fake_run(command, **_kwargs):
        Path(command[-1]).write_bytes(b"encoded")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(media.subprocess, "run", fake_run)

    result = media.convert(source, output_root)

    assert result == (tmp_path / "outputs" / "video_viewer_media" / "rendered" / "clip.mp4").resolve()
    assert result.read_bytes() == b"encoded"


def test_iter_video_paths_returns_sorted_mp4s_recursively(tmp_path):
    video_dir = tmp_path / "videos"
    nested = video_dir / "nested"
    nested.mkdir(parents=True)
    (video_dir / "b.mp4").touch()
    (nested / "a.mp4").touch()
    (video_dir / "ignore.avi").touch()

    assert list(media.iter_video_paths(video_dir)) == [
        (video_dir / "b.mp4").resolve(),
        (nested / "a.mp4").resolve(),
    ]
