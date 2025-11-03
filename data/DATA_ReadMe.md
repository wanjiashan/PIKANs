**DATA README**

**Dataset Overview**

This repository includes four open-pit mine slope monitoring datasets. Each entry lists location, extents, bench/slope geometry, monitoring modality, spatial coverage, and temporal sampling.

**1) Dexing Open-Pit Copper Mine Landslide (Jiangxi, China)**

- **Location:** Dexing City, Jiangxi Province, China  
    (E 117°37′00″ – 117°48′00″, N 28°59′00″ – 29°05′00″)
- **Pit extents:** 6.6 km (E–W) × 2.2 km (N–S), max depth 420 m; bench height 9–32 m; slope angle 45°–75°
- **Event:** north-slope landslide on 2018-05-05 (~1.5×10⁴ m³)
- **Monitoring:** GB-SAR deployed during the event; **sampling** every 7 min from 2018-05-04 14:24 to 2018-05-05 14:54 (210 epochs)
- **Notes:** time series include cumulative displacement and derived velocity (mm, mm/min)

**2) Xinjing Open-Pit Coal Mine Landslides (Alxa League, Inner Mongolia, China)**

- **Location:** (E 105°38′43″ – 105°40′03″, N 37°58′09″ – 37°58′49″)
- **Event:** two successive landslides on 2023-02-22 (~7.56×10⁶ m³ total)
- **Pre-failure indicators:** surface cracking, sporadic rockfalls, localized collapses
- **Monitoring / reconstruction:** Sentinel-1 high-resolution SAR; SBAS time series reconstructed via Google Earth Engine, **temporal span** 2021-01 to 2023-02 (orbit-driven revisit interval; see per-scene metadata)
- **Notes:** time series include cumulative displacement and derived velocity (mm, mm/min)

**3) Baijialiang Open-Pit Coal Mine (Ordos, Inner Mongolia, China)**

- **Location:** Nalin Taohai Town, Yijinhuoluo Banner, Ordos, Inner Mongolia, China  
    (E 110°14′14″ – 110°15′07″, N 39°31′34″ – 39°31′53″)
- **Pit extents (Dec 2024 UAV survey):** length 2.1 km, width 0.5 km, depth 70 m, area 0.66 km²
- **Bench / slope:** single bench height 8–10 m; overall slope angle 11°–34°
- **Monitoring:** GB-SAR real-time monitoring deployed; **coverage** ≈ 0.42 km²
- **Temporal coverage:** **sampling** every 10 min, since Oct 2024 (continuous acquisition; see per-file metadata for exact cadence)

**4) Hongjingta Open-Pit Coal Mine (Ordos, Inner Mongolia, China)**

- **Location:** Narison Town, Jungar Banner, Ordos, Inner Mongolia, China  
    (E 110°33′13″ – 110°33′55″, N 39°28′12″ – 39°29′00″)
- **Pit extents (Dec 2024 UAV survey):** length 2.5 km, width 1.7 km, depth 130 m, area 2.63 km²
- **Bench / slope:** single bench height 9–15 m; overall slope angle 17°–49°
- **Monitoring:** GB-SAR real-time monitoring on the north sector; **coverage** ≈ 0.67 km²
- **Temporal coverage:** **sampling** every 10 min, since Oct 2024 (continuous acquisition; see per-file metadata for exact cadence)

**File Structure & Conventions**

- **Units:** displacement = mm; velocity = mm/min (Dexing, GB-SAR); SBAS displacement = mm (Xinjing).
- **Time:** all timestamps in local time with ISO-8601 strings; epoch is a zero-based sequential index.
- **Coordinates:** WGS-84 geographic (degree-minute-second ranges above); per-file projected CRS (if any) is declared in metadata.

**Citation**

If you use these datasets, please cite the accompanying manuscript and this repository. Example:

“This work uses GB-SAR and SBAS datasets from the Baijialiang, Hongjingta, Dexing, and Xinjing open-pit mines (HFUT-OP360 Datasets).