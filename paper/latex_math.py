# -*- coding: utf-8 -*-
"""LaTeX → OMML (Word native equation) converter for python-docx.

Pipeline: LaTeX → MathML (latex2mathml) → OMML (Microsoft MML2OMML.XSL).
The result is a <m:oMath> element that Word renders as a real equation.
"""
from pathlib import Path
from xml.etree import ElementTree as ET

import latex2mathml.converter
from lxml import etree

XSL_PATH = Path(r"C:\Program Files\Microsoft Office"
                r"\root\Office16\MML2OMML.XSL")

_xslt = etree.XSLT(etree.parse(str(XSL_PATH)))

# OMML namespace
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def latex_to_omml(latex_str, display=False):
    """Convert LaTeX to an OMML element usable in python-docx."""
    mathml = latex2mathml.converter.convert(latex_str,
                                            display="block" if display else "inline")
    # parse MathML with lxml
    mathml_tree = etree.fromstring(mathml.encode())
    # apply Microsoft's XSLT
    omml_tree = _xslt(mathml_tree)
    # get the root <m:oMath> element
    root = omml_tree.getroot()
    # find all m:oMath elements
    omath_elements = root.findall(f"{{{M_NS}}}oMath")
    if omath_elements:
        return omath_elements[0]
    # sometimes the root itself is m:oMathPara wrapping m:oMath
    opara = root.findall(f"{{{M_NS}}}oMathPara")
    if opara:
        omath_elements = opara[0].findall(f"{{{M_NS}}}oMath")
        if omath_elements:
            return omath_elements[0]
    return root


def add_math_run(paragraph, latex_str, display=False):
    """Append a math run to an existing paragraph."""
    omml = latex_to_omml(latex_str, display)
    # convert lxml element to a string, then re-parse as python-docx element
    omml_str = etree.tostring(omml)
    # parse into python-docx's XML tree
    from docx.oxml import parse_xml
    # need namespace declarations
    nsmap = f'xmlns:m="{M_NS}" xmlns:w="{W_NS}"'
    wrapped = omml_str.decode()
    if 'xmlns:m' not in wrapped:
        # inject namespace into root tag
        wrapped = wrapped.replace('<m:oMath',
                                  f'<m:oMath {nsmap}', 1)
    el = parse_xml(wrapped)
    paragraph._p.append(el)
    return paragraph


def math_paragraph(doc, latex_str, number, display=True):
    """Create a centered paragraph with a display equation and its number."""
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format
    pf.line_spacing = 1.3
    pf.space_before = Pt(4)
    pf.space_after = Pt(6)

    # add the math
    add_math_run(p, latex_str, display=display)

    # add equation number (right-aligned via tab)
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    run = p.add_run("        (" + str(number) + ")")
    run.font.name = "Times New Roman"
    run.font.size = Pt(11)

    return p
