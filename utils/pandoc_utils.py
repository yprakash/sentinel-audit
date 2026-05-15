from pathlib import Path

import pypandoc


class PandocError(Exception):
    # # Later extend it like:
    # def __init__(self, message: str, stdout: str | None = None, stderr: str | None = None):
    #     super().__init__(message)
    #     self.stdout = stdout
    #     self.stderr = stderr
    pass


def markdown_to_pdf(
        markdown_path: str | Path,
        pdf_path: str | Path,
        *,
        toc: bool = True,
        number_sections: bool = True,
        pdf_engine: str = "xelatex",
) -> Path:
    markdown_path = Path(markdown_path)
    pdf_path = Path(pdf_path)

    extra_args = [
        f"--pdf-engine={pdf_engine}",
    ]

    if toc:
        extra_args.append("--toc")
    if number_sections:
        extra_args.append("--number-sections")

    try:
        pypandoc.convert_file(
            str(markdown_path),
            to="pdf",
            outputfile=str(pdf_path),
            extra_args=extra_args,
        )
    except RuntimeError as e:
        raise PandocError(str(e)) from e

    return pdf_path
