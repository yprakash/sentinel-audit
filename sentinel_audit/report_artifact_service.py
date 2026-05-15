import logging
import time
from pathlib import Path

from sentinel_audit.llm_outputs import AuditReport
from utils.jinja_utils import render_to_file
from utils.pandoc_utils import markdown_to_pdf

logger = logging.getLogger(__name__)


def generate_audit_artifacts(
        report: AuditReport,
        output_dir: str | Path,
        template_name: str = "audit_report.md.j2",
) -> dict[str, Path]:
    output_dir = Path(output_dir)

    markdown_paths = [
        output_dir / "audit_report.md",
        output_dir / f"audit_report_{int(time.time())}.md"
    ]
    pdf_paths = [
        output_dir / "audit_report.pdf",
        output_dir / f"audit_report_{int(time.time())}.pdf"
    ]

    for markdown_path, pdf_path in zip(markdown_paths, pdf_paths):
        path1 = render_to_file(
            template_name=template_name,
            context={"report": report.model_dump()},
            output_path=markdown_path,
        )
        path2 = markdown_to_pdf(
            markdown_path=markdown_path,
            pdf_path=pdf_path,
        )
        logger.info("Generated audit report for template %s written to %s and %s",
                    template_name, path1, path2)

    return {
        "markdown": markdown_paths[0],
        "pdf": pdf_paths[0],
    }
