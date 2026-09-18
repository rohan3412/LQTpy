# Export ggseg polygon geometry (DK, Schaefer-100 7-network) to CSV for the
# Python renderer. Run once (build-time): Rscript tools/export_ggseg_geometry.R
#
# Output: lqtpy/data/ggseg_<atlas>.csv.gz with columns
#   region, hemi, view, group, subgroup, x, y
# `region` matches lqtpy's atlas RegionName (DK: e.g. "fusiform_left" stripped
# to "fusiform" + hemi column; Schaefer: e.g. 7Networks_LH_Vis_1).
# group/subgroup separate polygons and holes within a region. view is one of
# lateral/medial/inferior/superior.
#
# NOTE: ggsegAAL is intentionally NOT exported. Its `aal` atlas ships with
# placeholder region labels (LH_region_N) and an unvalidated volume->surface
# mapping (see the package's data-raw/create_aal1.R, "NOT YET IMPLEMENTED"), so
# a damage map built from it would be plausible-looking but wrong.

options(warn = 1)
suppressMessages(library(ggseg))

outdir <- file.path("lqtpy", "data")
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

export_atlas <- function(atlas_obj, name, strip_prefix = "") {
  g <- atlas_obj$data$geom       # brain_polygons: columns label, geometry(list)
  parts <- vector("list", length(g$geometry))
  for (i in seq_along(g$geometry)) {
    lab <- g$label[i]
    poly <- as.data.frame(g$geometry[[i]])
    region <- sub("^[lr]h_", "", lab)
    if (nzchar(strip_prefix)) region <- sub(strip_prefix, "", region)
    poly$region <- region
    poly$hemi   <- ifelse(startsWith(lab, "lh_"), "left", "right")
    parts[[i]] <- poly[, c("region", "hemi", "view", "group", "subgroup", "x", "y")]
  }
  out <- do.call(rbind, parts)
  out$x <- round(out$x, 2)
  out$y <- round(out$y, 2)
  f <- file.path(outdir, paste0("ggseg_", name, ".csv.gz"))
  con <- gzfile(f, "w")
  write.csv(out, con, row.names = FALSE)
  close(con)
  cat("wrote", f, "\n rows:", nrow(out),
      " regions:", length(unique(out$region)),
      " views:", paste(unique(out$view), collapse = "/"), "\n")
}

suppressMessages(library(ggsegSchaefer))
export_atlas(schaefer7_100(), "schaefer100")

# dk() region labels are already bare names (e.g. "bankssts"); lqtpy's DK
# label table uses "bankssts_left"/"bankssts_right", so no prefix to strip.
export_atlas(dk(), "dk")
