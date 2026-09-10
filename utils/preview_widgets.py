"""Reusable image preview widgets for workshop notebooks."""

import asyncio
from base64 import b64encode
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from html import escape

import ipywidgets as widgets


_PREVIEW_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="workshop-preview")


_PREVIEW_STYLE = """<style>
.preview-index-input input[type="number"] {
    box-sizing: border-box !important;
    width: 72px !important;
    height: 32px !important;
    min-height: 32px !important;
    padding: 0 10px !important;
    border: 1px solid var(--jp-border-color2, #b8bec7) !important;
    border-radius: 16px !important;
    background: var(--jp-layout-color0, #ffffff) !important;
    color: var(--jp-ui-font-color1, #24292f) !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.08);
    text-align: center !important;
    font-weight: 500;
    font-variant-numeric: tabular-nums;
    appearance: textfield;
    -moz-appearance: textfield;
    transition: border-color 120ms ease, box-shadow 120ms ease;
}
.preview-index-input input[type="number"]:hover {
    border-color: var(--jp-border-color1, #8c95a3) !important;
}
.preview-index-input input[type="number"]:focus {
    outline: none !important;
    border-color: #ed1c24 !important;
    box-shadow: 0 0 0 2px rgba(237, 28, 36, 0.16) !important;
}
.preview-index-input input[type="number"]::-webkit-inner-spin-button,
.preview-index-input input[type="number"]::-webkit-outer-spin-button {
    margin: 0;
    appearance: none;
    -webkit-appearance: none;
}
.preview-result-badge {
    display: inline-block;
    margin-left: 8px;
    padding: 2px 8px;
    border-radius: 999px;
    background: rgba(237, 28, 36, 0.10);
    color: #c8171e;
    font-size: 0.85em;
    font-weight: 600;
    white-space: nowrap;
}
.preview-slider-axis {
    position: relative;
    box-sizing: border-box;
    width: calc(100% - 16px);
    height: 19px;
    margin: -1px 8px 0;
    color: var(--jp-ui-font-color2, #5f6368);
    font-size: 11px;
    line-height: 14px;
    font-variant-numeric: tabular-nums;
    user-select: none;
}
.preview-slider-tick {
    position: absolute;
    top: 0;
    white-space: nowrap;
}
.preview-slider-tick::before {
    content: "";
    display: block;
    width: 1px;
    height: 4px;
    margin: 0 auto 1px;
    background: var(--jp-border-color2, #b8bec7);
}
</style>"""


def _slider_axis(count: int) -> str:
    """Return a sparse, evenly spaced index axis for a preview slider."""
    if count == 1:
        return (
            '<div class="preview-slider-axis" aria-hidden="true">'
            '<span class="preview-slider-tick" style="left:50%;transform:translateX(-50%)">1</span></div>'
        )
    tick_count = count if count <= 10 else 6 if count <= 50 else 5
    values = sorted({round(1 + step * (count - 1) / (tick_count - 1)) for step in range(tick_count)})
    ticks = []
    for position, value in enumerate(values):
        left = 100 * (value - 1) / (count - 1)
        shift = "0" if position == 0 else "-100%" if position == len(values) - 1 else "-50%"
        ticks.append(
            f'<span class="preview-slider-tick" style="left:{left:.3f}%;transform:translateX({shift})">'
            f"{value}</span>"
        )
    return '<div class="preview-slider-axis" aria-hidden="true">' + "".join(ticks) + "</div>"


def image_slider_panel(
    title: str,
    items: Sequence[str],
    initial: str,
    render: Callable[[str], bytes | tuple[bytes, str]],
) -> widgets.VBox:
    """Build a responsive image panel with linked slider and number input."""
    heading = widgets.HTML(layout=widgets.Layout(width="100%"))
    image = widgets.HTML(layout=widgets.Layout(width="100%", overflow="hidden"))
    slider = widgets.IntSlider(
        value=items.index(initial) + 1,
        min=1,
        max=len(items),
        readout=False,
        continuous_update=False,
        layout=widgets.Layout(width="100%"),
    )
    number = widgets.BoundedIntText(
        value=slider.value,
        min=slider.min,
        max=slider.max,
        continuous_update=False,
        layout=widgets.Layout(width="72px", height="32px"),
    )
    number.add_class("preview-index-input")
    generation = 0
    pending = None

    def render_item(index: int) -> tuple[str, bytes, str]:
        item = items[index - 1]
        rendered = render(item)
        payload, detail = rendered if isinstance(rendered, tuple) else (rendered, "")
        return item, payload, detail

    def show_result(requested: int, index: int, result: tuple[str, bytes, str]) -> None:
        if requested != generation:
            return
        item, payload, detail = result
        badge = f'<span class="preview-result-badge">{escape(detail)}</span>' if detail else ""
        heading.value = _PREVIEW_STYLE + (
            '<div style="width:100%;text-align:center">'
            f"<b>{escape(title)}</b> — <code>{escape(item)}</code> "
            f"({index}/{len(items)}){badge}</div>"
        )
        encoded = b64encode(payload).decode()
        image.value = (
            f'<img src="data:image/jpeg;base64,{encoded}" '
            'style="display:block;width:100%;max-width:100%;height:auto;'
            'aspect-ratio:16/9;object-fit:contain;background:#202020">'
        )

    def show_error(requested: int, error: Exception) -> None:
        if requested == generation:
            heading.value = _PREVIEW_STYLE + (
                '<div style="width:100%;text-align:center;color:#c8171e">'
                f"Preview failed: {escape(str(error))}</div>"
            )

    def schedule_update(_=None) -> None:
        nonlocal generation, pending
        generation += 1
        requested = generation
        index = slider.value
        if pending is not None and not pending.done():
            pending.cancel()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            show_result(requested, index, render_item(index))
            return

        pending = _PREVIEW_EXECUTOR.submit(render_item, index)

        def finished(future) -> None:
            if future.cancelled():
                return
            try:
                result = future.result()
            except Exception as error:
                loop.call_soon_threadsafe(show_error, requested, error)
            else:
                loop.call_soon_threadsafe(show_result, requested, index, result)

        pending.add_done_callback(finished)

    slider.observe(schedule_update, names="value")
    slider_axis = widgets.HTML(
        value=_slider_axis(len(items)),
        layout=widgets.Layout(width="100%", height="19px", overflow="hidden"),
    )
    slider_stack = widgets.VBox(
        [slider, slider_axis],
        layout=widgets.Layout(width="calc(100% - 84px)", min_width="0", overflow="visible"),
    )
    controls = widgets.HBox(
        [slider_stack, number],
        layout=widgets.Layout(
            width="100%",
            justify_content="space-between",
            align_items="flex-start",
            overflow="hidden",
        ),
    )
    controls._value_link = widgets.link((slider, "value"), (number, "value"))
    show_result(generation, slider.value, render_item(slider.value))
    panel = widgets.VBox(
        [heading, image, controls],
        layout=widgets.Layout(max_width="640px", min_width="0", flex="1 1 0", overflow="hidden"),
    )
    panel._preview_slider = slider
    return panel


def side_by_side_previews(left: widgets.VBox, right: widgets.VBox, linked: bool = False) -> widgets.HBox:
    """Place two responsive preview panels side by side without overflow."""
    left.layout.width = right.layout.width = "calc(50% - 20px)"
    preview = widgets.HBox(
        [left, right],
        layout=widgets.Layout(
            width="100%",
            max_width="1320px",
            flex_flow="row nowrap",
            justify_content="space-between",
            align_items="flex-start",
            overflow="hidden",
        ),
    )
    if linked:
        preview._value_link = widgets.link((left._preview_slider, "value"), (right._preview_slider, "value"))
    return preview
