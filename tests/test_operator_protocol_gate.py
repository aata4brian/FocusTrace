from pathlib import Path


def test_gate_controls_are_not_blocked_by_action_capture():
    text=Path('frontend/public/operator-protocol.js').read_text(encoding='utf-8')
    assert "if (!button || button.closest('#tf-research-gate')) return;" in text
    assert "if (b.closest('#tf-research-gate')) return false;" in text


def test_post_session_completion_is_persisted():
    text=Path('frontend/public/operator-protocol.js').read_text(encoding='utf-8')
    assert "tracefokus-protocol-post-done:" in text
    assert "state.state === 'FINISHED'" in text and "postDone() ? renderReady() : renderPost()" in text
