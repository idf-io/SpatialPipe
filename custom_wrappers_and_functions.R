# Sanity check
cat("Script `custom_wrappers_and_functions.R` loaded!\n")

# Sanity test function
scrip_status <- function() {
    cat("Script `custom_wrappers_and_functions.R` loaded!\n")
}

# Original MISTy
get_weight <- function(family = c(
    "gaussian", "exponential",
    "linear", "constant"
),
distances, parameter, zoi) {
    expr.family <- match.arg(family)
    
    distances[distances < zoi] <- Inf
    dim.orig <- dim(distances)
    
    switch(expr.family,
           "gaussian" = {
               exp(-distances^2 / parameter^2)
           },
           "exponential" = {
               exp(-distances / parameter)
           },
           "linear" = {
               weights <- pmax(0, 1 - distances / parameter)
               dim(weights) <- dim.orig
               weights
           },
           "constant" = {
               weights <- as.numeric(!is.infinite(distances))
               dim(weights) <- dim.orig
               weights
           }
    )
}


# Para-view modification to be able to change the data to which it is applied. AKA not the same as the default (intra-view)
add_paraview_custom <- function(current.views, extra.views, positions, l, zoi = 0,
                         family = c(
                             "gaussian", "exponential",
                             "linear", "constant"
                         ),
                         approx = 1, nn = NULL, prefix = "",
                         cached = FALSE, verbose = TRUE) {
    dists <- distances::distances(as.data.frame(positions))
    expr <- extra.views[[1]][["data"]]#current.views[["intraview"]][["data"]]
    
    cache.location <- R.utils::getAbsolutePath(paste0(
        ".misty.temp", .Platform$file.sep,
        current.views[["misty.uniqueid"]]
    ))
    
    para.cache.file <- paste0(
        cache.location, .Platform$file.sep,
        "para.view.", l, ".rds"
    )
    
    if (cached && !dir.exists(cache.location)) {
        dir.create(cache.location, recursive = TRUE, showWarnings = TRUE)
    }
    
    if (match.arg(family) == "constant" & is.numeric(l) & is.null(nn)) {
        nn <- round(l)
    }
    
    if (cached & file.exists(para.cache.file)) {
        para.view <- readr::read_rds(para.cache.file)
    } else {
        if (is.null(nn)) {
            if (approx == 1) {
                if (verbose) message("\nGenerating paraview")
                para.view <- seq(nrow(expr)) %>%
                    furrr::future_map_dfr(~ data.frame(t(colSums(expr[-.x, ] *
                                                                     get_weight(family, dists[, .x][-.x], l, zoi)))),
                                          .options = furrr::furrr_options(packages = "distances"),
                                          .progress = verbose
                    )
            } else {
                if (approx < 1) approx <- base::round(approx * ncol(dists))
                
                if (verbose) {
                    message("\nApproximating RBF matrix using the Nystrom method")
                }
                
                assertthat::assert_that(requireNamespace("MASS", quietly = TRUE),
                                        msg = "The package MASS is required to approximate the paraview using
          the Nystrom method."
                )
                
                # single Nystrom approximation expert, given RBF with parameter l
                s <- sort(sample.int(n = ncol(dists), size = approx))
                C <- get_weight(family, dists[, s], l, zoi)
                # pseudo inverse of W
                W.plus <- MASS::ginv(C[s, ])
                # return Nystrom list
                K.approx <- list(s = s, C = C, W.plus = W.plus)
                para.view <- seq(nrow(expr)) %>%
                    furrr::future_map_dfr(~ data.frame(t(colSums(expr[-.x, ] *
                                                                     sample_nystrom_row(K.approx, .x)[-.x]))),
                                          .progress = verbose
                    )
            }
        } else {
            if (verbose) {
                message(
                    "\nGenerating paraview using ", nn,
                    " nearest neighbors per unit"
                )
            }
            para.view <- seq(nrow(expr)) %>%
                furrr::future_map_dfr(function(rowid) {
                    knn <- distances::nearest_neighbor_search(dists, nn + 1,
                                                              query_indices = rowid
                    )[-1, 1]
                    data.frame(t(colSums(expr[knn, ] *
                                             get_weight(family, dists[knn, rowid], l, zoi))))
                },
                .options = furrr::furrr_options(packages = "distances"),
                .progress = verbose
                )
        }
        if (cached) readr::write_rds(para.view, para.cache.file)
    }
    
    return(current.views %>% add_views(create_view(
        paste0("paraview.", l),
        para.view %>% dplyr::rename_with(~paste0(prefix, .x)),
        paste0("para.", l)
    )))
}