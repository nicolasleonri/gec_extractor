# install.packages("dplyr")
library(dplyr)

############# Functions ########################
fix_invalid_dates <- function(date_vector, format = "%d/%m/%Y") {
  # Fix invalid dates by going back one day
  parsed_dates <- as.Date(date_vector, format = format)
  failed_indices <- which(is.na(parsed_dates))
  if (length(failed_indices) == 0) {
    cat("No invalid dates found!\n")
    return(parsed_dates)
  }
  cat("Found", length(failed_indices), "invalid dates to fix\n")
  for (i in failed_indices) {
    original_date <- date_vector[i]
    cat("Processing:", original_date)
    if (grepl("^\\d{1,2}/\\d{1,2}/\\d{4}$", original_date)) {
      parts <- strsplit(original_date, "/")[[1]]
      day <- as.numeric(parts[1])
      month <- as.numeric(parts[2])
      year <- as.numeric(parts[3])
      
      # Instead of just going back one day, find the last valid day of the month
      # Get the maximum valid day for this month/year combination
      max_day_in_month <- as.numeric(format(
        as.Date(paste(year, month + 1, "01", sep = "-")) - 1, 
        "%d"
      ))
      # Handle December (month 12)
      if (month == 12) {
        max_day_in_month <- as.numeric(format(
          as.Date(paste(year + 1, "01", "01", sep = "-")) - 1, 
          "%d"
        ))
      }
      # Set the day to the maximum valid day for that month
      new_day <- max_day_in_month
      # Create the corrected date string
      corrected_date_str <- sprintf("%02d/%02d/%04d", new_day, month, year)
      
      # Try to parse the corrected date
      corrected_date <- as.Date(corrected_date_str, format = format)
      
      if (!is.na(corrected_date)) {
        parsed_dates[i] <- corrected_date
        cat(" -> Fixed to:", corrected_date_str, "\n")
      } else {
        cat(" -> Still invalid after correction\n")
      }
    } else {
      cat(" -> Unrecognized format, skipping\n")
    }
  }
  return(parsed_dates)
}

preprocess_df <- function(input) {
  output <- read.csv(input, header=TRUE, sep=";", dec = ".", fileEncoding = "UTF-8")
  output$date_fixed <- fix_invalid_dates(output$date, "%d/%m/%Y")
  output$date <- output$date_fixed
  output$date_fixed <- NULL
  output$newspaper <- as.factor(output$newspaper)
  output$model_0_topic <- as.factor(output$model_0_topic)
  output$model_0_topic_label <- as.factor(output$model_0_topic_label)
  output$model_1_topic <- as.factor(output$model_1_topic)
  output$model_1_topic_label <- as.factor(output$model_1_topic_label)
  output$model_2_topic <- as.factor(output$model_2_topic)
  output$model_2_topic_label <- as.factor(output$model_2_topic_label)
  return(output)
}

filter_by_keywords <- function(df, keywords) {
  pattern <- paste(keywords, collapse = "|")
  mask <- grepl(pattern, df$model_0_topic_label, ignore.case = TRUE) |
    grepl(pattern, df$model_1_topic_label, ignore.case = TRUE) |
    grepl(pattern, df$model_2_topic_label, ignore.case = TRUE)
  df[mask, ]
}

############### CODE ####################
# Read csv files
csv_files <- list.files(path = "./data", pattern = "*.csv", full.names = TRUE)
# Preprocess and reads csv files
df_list <- list()
for (i in seq_along(csv_files)) {
  cat("Preprocessing file", i, ":", basename(csv_files[i]), "\n")
  df_list[[i]] <- preprocess_df(csv_files[i])
}
# Initialize with the first dataframe
combined_df <- df_list[[1]]
# Get model prefixes dynamically (any column ending with "_topic")
model_cols <- grep("_topic$", names(combined_df), value = TRUE)
model_prefixes <- sub("_topic$", "", model_cols)
# Create mapping list: one mapping df per model
mapping_list <- lapply(model_prefixes, function(prefix) {
  unique(combined_df[c(paste0(prefix, "_topic"), paste0(prefix, "_topic_label"))])
})
names(mapping_list) <- model_prefixes
for (i in 2:length(df_list)) {
  current_df <- df_list[[i]]
  for (prefix in model_prefixes) {
    topic_col <- paste0(prefix, "_topic")
    label_col <- paste0(prefix, "_topic_label")
    # Current mapping for this model
    current_mapping <- mapping_list[[prefix]]
    # --- Find new labels not yet in mapping
    new_labels <- !current_df[[label_col]] %in% current_mapping[[label_col]]
    if (any(new_labels)) {
      new_rows <- unique(current_df[new_labels, c(topic_col, label_col)])
      # Convert to numeric safely
      new_rows[[topic_col]] <- as.numeric(as.character(new_rows[[topic_col]]))
      combined_max <- max(as.numeric(as.character(combined_df[[topic_col]])), na.rm = TRUE)
      # Shift topics so they start after existing ones
      new_rows[[topic_col]] <- new_rows[[topic_col]] + combined_max
      # Convert back to factor to match original type
      new_rows[[topic_col]] <- as.factor(new_rows[[topic_col]])
      # Append to mapping
      current_mapping <- rbind(current_mapping, new_rows)
    }
    # --- Remap topics in current_df to use updated mapping
    match_idx <- match(current_df[[label_col]], current_mapping[[label_col]])
    current_df[[topic_col]] <- current_mapping[[topic_col]][match_idx]
    # Update mapping_list with the new mapping
    mapping_list[[prefix]] <- current_mapping
  }
  # Update df_list and combined_df
  df_list[[i]] <- current_df
  combined_df <- rbind(combined_df, current_df)
}

################ EXTRACTION ######################
keywords_economia_informal <- c(
  # Core terms - broad matching
  "informal", "ambulant", "formal",
  
  # Key phrases - informal economy
  "economia informal", "economía informal", 
  "economia sumergida", "economía sumergida",
  "economia subterránea", "economía subterránea",
  "economia popular", "economía popular",
  "economia en la sombra", "economía en la sombra",
  "economia de subsistencia", "economía de subsistencia",
  "sector informal", "mercado informal", "actividad informal", "comercio informal",
  
  # Employment terms
  "empleo informal", "trabajo informal", "trabajo no regulado", "empleo no declarado",
  "trabajador informal", "trabajadores informales",
  "trabajo diario", "autoempleo", "cuenta propia", "por cuenta propia",
  "subempleo", "sin contrato", "no registrado", "sin beneficios", "sin protección",
  "familiar no remunerado", "familiares no remunerados", "familiar auxiliar",
  "trabajo doméstico",
  
  # Formalization & evasion ("evasión", "evasion", "evadir", "eludir",)
  "formalización", "formalizacion", "formaliza", 
  
  # Peruvian context - specific locations & activities ("mototaxi", "combi",)
  "Gamarra", "La Parada",
  "cachuelo", "vendedor callejero",
  
  # Labor conditions & characteristics
  "precariedad laboral", "mujer informal", "mujeres informales",
  "bajos ingresos", "baja productividad",
  "inseguridad económica", "inseguridad economica",
  
  # Policy & institutional
  "política de formalización", "politica de formalizacion",
  "políticas de formalización", "politicas de formalizacion",
  "formalización laboral", "formalizacion laboral",
  "registro laboral", "inspección laboral", "inspeccion laboral",
  "micro y pequeñas empresas", "MYPE",
  
  # International references
  "International Labour Organisation", "OECD", "Keith Hart",
  "Organización internacional del trabajo", "Organizacion internacional del trabajo"
)
keywords_modernizante <- c(
  "sector dual", "Arthur Lewis", "premoderno", "subsistencia", "industrialización", "industrializacion",
  "desarrollo económico", "desarrollo economico", "absorción", "absorcion", "sector capitalista",
  "etapas de desarrollo", "transición", "transicion", "formalización", "formalizacion",
  "progreso", "modernización", "modernizacion", "residuo", "atraso"
)
keywords_estructuralista <- c(
  "marxismo", "capitalismo", "exclusión", "exclusion", "salarios bajos", "competencia laboral",
  "explotación", "explotacion", "plusvalía", "plusvalia", "reserva de mano de obra",
  "desigualdad estructural", "sistema económico", "sistema economico", "clase trabajadora",
  "precariado", "explotados", "informalidad estructural",
  "dependencia", "periferia", "centro", "migración rural-urbana", "migracion rural-urbana",
  "precarización", "precarizacion", "deslocalización", "deslocalizacion",
  "externalización", "externalizacion", "subcontratación", "subcontratacion"
)
keywords_neoliberal <- c(
  "Hernando de Soto", "burocracia", "regulaciones", "impuestos", "intervención estatal", "intervencion estatal",
  "flexibilidad", "autonomía", "autonomia", "libre mercado", "costos de formalización", "costos de formalizacion",
  "trámites", "tramites", "emprendimiento", "libertad económica", "libertad economica",
  "deregulación", "desregulación", "desregulacion", "mercado libre",
  "barreras regulatorias", "racionalidad", "elección individual", "eleccion individual",
  "evitar impuestos", "informalidad voluntaria"
)
keywords_posmoderna <- c(
  "redes de solidaridad", "antropología", "antropologia", "cultura", "reciprocidad", "comunidad",
  "capital social", "trueque", "mercados populares", "identidad", "tradición", "tradicion",
  "resistencia cultural", "economía alternativa", "economia alternativa",
  "redistribución", "redistribucion", "cooperación", "cooperacion",
  "valores comunitarios", "informalidad cultural", "prácticas locales", "practicas locales",
  "solidaridad", "ayni", "minka", "minga"
)
keywords_voluntarista <- c(
  "evasión", "evasion", "competencia desleal", "regulaciones ineficientes", "beneficios",
  "maximización", "maximizacion", "estrategia", "ventaja competitiva", "mercado libre",
  "abuso de controles", "ineficiencia estatal", "opción racional", "opcion racional",
  "beneficio individual", "eludir normas", "informalidad estratégica", "informalidad estrategica",
  "rentabilidad", "fraude", "subdeclaración", "subdeclaracion", "economía ilegal", "economia ilegal"
)
### Apply for each conceptual group ###
economia_informal_df <- filter_by_keywords(combined_df, keywords_economia_informal)
perspectiva_modernizante_df <- filter_by_keywords(economia_informal_df, keywords_modernizante)
perspectiva_estructuralista_df <- filter_by_keywords(economia_informal_df, keywords_estructuralista)
perspectiva_neoliberal_df <- filter_by_keywords(economia_informal_df, keywords_neoliberal)
perspectiva_posmoderna_df <- filter_by_keywords(economia_informal_df, keywords_posmoderna)
perspectiva_voluntarista_df <- filter_by_keywords(economia_informal_df, keywords_voluntarista)

### Extract randomized/stratified samples ###
set.seed(123)  # For reproducibility

# Considers newspapers and years
sample_df <- economia_informal_df %>%
  mutate(year = format(date, "%Y")) %>%  # Extract year from date
  group_by(newspaper, year) %>%
  slice_sample(prop = 1125 / nrow(economia_informal_df)) %>%
  ungroup()

table(economia_informal_df$newspaper)
table(sample_df$newspaper)

# Shuffle the sample_df
sample_df <- sample_df %>%
  slice_sample(n = nrow(sample_df))

### Save as csv
write.csv(sample_df, file = "./sample_df.csv", row.names = FALSE)
write.csv(economia_informal_df, file = "./economia_informal_df.csv", row.names = FALSE)
write.csv(perspectiva_modernizante_df, file = "./perspectiva_modernizante_df.csv", row.names = FALSE)
write.csv(perspectiva_estructuralista_df, file = "./perspectiva_estructuralista_df.csv", row.names = FALSE)
write.csv(perspectiva_neoliberal_df, file = "./perspectiva_neoliberal_df.csv", row.names = FALSE)
write.csv(perspectiva_posmoderna_df, file = "./perspectiva_posmoderna_df.csv", row.names = FALSE)
write.csv(perspectiva_voluntarista_df, file = "./perspectiva_voluntarista_df.csv", row.names = FALSE)
