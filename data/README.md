# Airline Delay Cause Dataset (`Airline_Delay_Cause.csv`)

## What this file contains

This dataset summarizes **U.S. flight arrival and delay performance** by:
- `year`
- `month`
- `carrier` (airline)
- `airport`

Each row is a **monthly aggregate** for one airline at one airport (not an individual flight record).

It includes:
- total arriving flights
- how many were delayed 15+ minutes
- cancellations and diversions
- delay causes (carrier, weather, NAS, security, late aircraft)
- total delay minutes and delay minutes by cause

## Why this data is useful

You can use this file to:
- compare carrier performance across airports
- find which delay causes are most common
- study seasonal delay patterns by month
- build models to predict delay volume/rate
- detect unusual delay patterns (anomalies)

## Column dictionary

- `year`: Year of the record.
- `month`: Month of the record (`1`-`12`).
- `carrier`: Airline carrier code.
- `carrier_name`: Airline name.
- `airport`: Airport code.
- `airport_name`: Airport name.
- `arr_flights`: Number of arriving flights.
- `arr_del15`: Number of arrivals delayed by 15 minutes or more.
- `carrier_ct`: Delay count attributed to the carrier.
- `weather_ct`: Delay count attributed to weather.
- `nas_ct`: Delay count attributed to NAS (National Airspace System).
- `security_ct`: Delay count attributed to security.
- `late_aircraft_ct`: Delay count attributed to late arriving aircraft.
- `arr_cancelled`: Number of cancelled arrivals.
- `arr_diverted`: Number of diverted arrivals.
- `arr_delay`: Total arrival delay minutes (all causes combined).
- `carrier_delay`: Arrival delay minutes attributed to carrier issues.
- `weather_delay`: Arrival delay minutes attributed to weather.
- `nas_delay`: Arrival delay minutes attributed to NAS issues.
- `security_delay`: Arrival delay minutes attributed to security issues.
- `late_aircraft_delay`: Arrival delay minutes attributed to late aircraft.

## Notes

- Delay cause counts may appear as decimals in some records because reporting is aggregated.
- Cause-specific counts and delay minutes are related but not identical (count of delayed flights vs. total minutes of delay).
