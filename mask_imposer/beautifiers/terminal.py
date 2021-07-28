from typing import Optional

from progressbar import ProgressBar


class TerminalProgressBar:  # pylint: disable=R0903
    """Represent terminal progressbar.

        Example:
            20% [#####===============>]
    """

    def __init__(self) -> None:
        self.terminal_bar: Optional[ProgressBar] = None

    def __call__(self, block_num: int, block_size: int, total_size: int) -> None:
        if not self.terminal_bar:
            self.terminal_bar = ProgressBar(maxval=total_size)
            self.terminal_bar.start()
        downloaded = block_num * block_size
        if downloaded < total_size:
            self.terminal_bar.update(downloaded)
        else:
            self.terminal_bar.finish()
