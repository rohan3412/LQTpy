"""Advice for clinicians, and an action name for a drawing tool's button.

A finding's ``suggestion`` stays as it was (some name lqtpy functions,
which suits programmers). This module adds, per finding id, advice in plain
words for the person fixing the mask, and a stable **action** name a tool
maps to its own control:

==================  =============================================  =========
action              what it means                                  qc.fix id
==================  =============================================  =========
remove_specks       delete stray single voxels / tiny bits         specks
remove_strokes      delete thin lines drawn by mistake             strokes
remove_edge         delete the piece touching the image edge       edge_artifact
remove_outside      delete voxels beyond the brain's edge          outside_brain
interpolate_gaps    fill the skipped slices between drawn ones     sparse_slices
complete_slices     fill the missing parts of partly drawn slices  partial_gaps
fill_holes          fill small holes inside the lesion             holes
fill_outline        fill an outline drawn without its inside       hollow
redraw              redraw the part named (no automatic fix)
review              look at it; may be fine (no automatic fix)
check_registration  check the mask's space / registration
check_side          check left and right
check_file          check the file itself (header, values)
==================  =============================================  =========

Names are part of the API: they are only ever added, never renamed.
Opt-in: ``finding.advice`` / ``finding.action``, :func:`advice_for`, or
``QCResult.to_dict(advice=True)``; nothing else changes.
"""
from __future__ import annotations

__all__ = ["ADVICE", "ACTION_FIX", "advice_for"]

#: finding id -> (action, advice in plain words)
ADVICE = {
    "sparse_slices": ("interpolate_gaps",
                      "Some slices between drawn ones were skipped. Draw on every slice through the "
                      "lesion, or fill the skipped slices by interpolation and check the result."),
    "partial_gaps": ("complete_slices",
                     "Some slices are drawn only in part. Complete the outline on those slices."),
    "holes": ("fill_holes", "There are small unfilled holes inside the lesion. Fill them in."),
    "hollow": ("fill_outline",
               "Only the outline was drawn. Fill the inside of the lesion on each slice."),
    "flat": ("redraw",
             "This part is drawn on a single slice, with no depth. Draw it on every slice it "
             "appears on."),
    "strokes": ("remove_strokes", "Thin lines look drawn by mistake. Erase them."),
    "specks": ("remove_specks", "There are stray voxels away from the lesion. Erase them."),
    "box": ("redraw",
            "A filled rectangle looks like a placeholder or a cropping box, not the lesion. "
            "Redraw that part along the lesion's edge."),
    "edge_artifact": ("remove_edge",
                      "A piece touches the edge of the image, which usually comes from "
                      "registration or export, not from drawing. Erase it."),
    "outside_brain": ("remove_outside",
                      "Part of the mask lies beyond the brain's edge. Erase what is outside, or "
                      "check the registration if a lot is outside."),
    "comb": ("review",
             "The border is ragged from slice to slice (comb-like). Smooth it if the lesion's "
             "real edge is smooth; it may be acceptable as it is."),
    "blocky": ("review",
               "The border climbs in steps, as if drawn on a coarse grid. Redraw at full "
               "resolution if the shape matters for the analysis."),
    "fragmentation": ("review",
                      "The lesion is split into very many small pieces. Check that they are "
                      "all lesion, not noise."),
    "plausibility": ("review", "Something about this lesion is unusual. Look at it before using it."),
    "misregistered": ("check_registration",
                      "The mask doesn't line up with the brain. Check that it was registered "
                      "to MNI space correctly."),
    "not_mni": ("check_registration",
                "The mask doesn't look like it is in MNI space. Register it to MNI first."),
    "inverted": ("check_side",
                 "Left and right may be swapped. Check the image's orientation."),
    "side_mismatch": ("check_side",
                      "Most of the lesion is on the other side from the one expected. Check left "
                      "and right, and the patient's details."),
    "header": ("check_file",
               "The file's position information is inconsistent. Re-export the mask from the "
               "drawing tool."),
    "input": ("check_file", "The file has a problem (empty, unreadable values). Check the file."),
    "values": ("check_file",
               "The mask has more than plain lesion / no-lesion values. Check that the right "
               "file was saved."),
}

#: action -> the :func:`lqtpy.qc.fix` id that does it (actions without one need a person).
ACTION_FIX = {"remove_specks": "specks", "remove_strokes": "strokes", "remove_edge": "edge_artifact",
              "remove_outside": "outside_brain", "interpolate_gaps": "sparse_slices",
              "complete_slices": "partial_gaps", "fill_holes": "holes", "fill_outline": "hollow"}
_ADDS = {"interpolate_gaps", "complete_slices", "fill_holes", "fill_outline"}


def advice_for(finding) -> dict:
    """``{"action", "text", "fix_id", "adds_voxels"}`` for a Finding (or its
    dict or id). Unknown ids get ``review`` and the finding's own message."""
    fid = finding if isinstance(finding, str) else (
        finding.get("id") if isinstance(finding, dict) else finding.id)
    action, text = ADVICE.get(fid, ("review", ""))
    if not text and not isinstance(finding, str):
        text = finding.get("message", "") if isinstance(finding, dict) else finding.message
    fix_id = ACTION_FIX.get(action)
    return {"action": action, "text": text, "fix_id": fix_id,
            "adds_voxels": action in _ADDS if fix_id else None}
