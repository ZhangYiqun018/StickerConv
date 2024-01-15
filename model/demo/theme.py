from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
from typing import Union, Iterable


class CustomTheme(Base):
    def __init__(
        self,
        primary_hue: Union[colors.Color, str] = colors.emerald,
        secondary_hue: Union[colors.Color, str] = colors.blue,
        neutral_hue: Union[colors.Color, str] = colors.slate,
        spacing_size: Union[sizes.Size, str] = sizes.spacing_md,
        radius_size: Union[sizes.Size, str] = sizes.radius_md,
        text_size: Union[sizes.Size, str] = sizes.text_lg,
        font: Union[fonts.Font, str, Iterable[Union[fonts.Font, str]]] = (
            fonts.GoogleFont("Alice"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono: Union[fonts.Font, str, Iterable[Union[fonts.Font, str]]] = (
            fonts.GoogleFont("Merriweather"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            body_background_fill="#ECF2F7",
            body_background_fill_dark="#191919",
            button_primary_background_fill="linear-gradient(90deg, *primary_300, *secondary_400)",
            button_primary_background_fill_hover="*primary_700",
            button_primary_text_color="white",
            button_primary_background_fill_dark="linear-gradient(90deg, *primary_600, *secondary_800)",
            slider_color="#4EACEF",
            slider_color_dark="#4EACEF",
            block_title_text_weight="600",
            block_title_text_size="*text_md",
            block_label_text_weight="600",
            block_label_text_size="*text_md",
            block_border_width="1px",
            block_shadow="#FFFFFF00",
            button_shadow="*shadow_drop_lg",
            button_large_padding="*spacing_lg calc(2 * *spacing_lg)",
        )