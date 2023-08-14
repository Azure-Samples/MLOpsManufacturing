# Symbol Hierarchy and Nomenclature

## Objective of Symbol Categorisation

Due to a more free form approach to the P&ID symbols, Industrial Process Control vendors represent the same symbol differently. Thus, to aid the user workflow in labelling P&IDs provided by various vendors , these 3 - level symbol hierarchy has been introduced as an example of best practices.

The user also has the provision to modify this hierarchy to a single or dual level (method to modify this hierarchy would be included in this document as part of further revisions).

This grouping of symbols would help the user to easily label, identify and modify the detected symbols. This also solves the problem of naming duplication of the detected symbols.

The 3 main categories are Instrument, Piping, and Equipment.

Possible drill down is shown here. For example:

```text
Equipment/
    Vessel/
        Actual equipment name
```

## Nomenclature

| **Symbol ID** | **Category** | **Sub-category** | **Display Label**                | **Concatenated unique string**                          |
|---------------|--------------|------------------|----------------------------------|---------------------------------------------------------|
| 1             | Instrument   | Valve            | Welded gate valve                | Instrument/Valve/Welded gate valve                    |
| 2             | Instrument   | Valve            | Plug valve                       | Instrument/Valve/Plug valve                           |
| 3             | Instrument   | Valve            | Globe valve NO                   | Instrument/Valve/Globe valve NO                       |
| 4             | Instrument   | Valve            | Gate valve NO                    | Instrument/Valve/Gate valve NO                        |
| 5             | Instrument   | Valve            | Ball valve                       | Instrument/Valve/Ball valve                           |
| 6             | Instrument   | Valve            | Butterfly Valve                  | Instrument/Valve/Butterfly Valve                      |
| 7             | Instrument   | Valve            | Manual gate valve                | Instrument/Valve/Manual gate valve                    |
| 8             | Instrument   | Valve            | Check valve                      | Instrument/Valve/Check valve                          |
| 9             | Instrument   | Valve            | Diaphragm valve                  | Instrument/Valve/Diaphragm valve                      |
| 10            | Instrument   | Valve            | Needle Valve                     | Instrument/Valve/Needle Valve                         |
| 11            | Instrument   | Valve            | Sealing gate valve               | Instrument/Valve/Sealing gate valve                   |
| 12            | Instrument   | Valve            | Gate valve NC                    | Instrument/Valve/Gate valve NC                        |
| 13            | Instrument   | Valve            | Globe valve NC                   | Instrument/Valve/Globe valve NC                       |
| 14            | Instrument   | Valve            | Control Valve                    | Instrument/Valve/Control Valve                        |
| 15            | Instrument   | Valve            | Rotary valve NO                  | Instrument/Valve/Rotary valve NO                      |
| 16            | Instrument   | Valve            | Rotary valve NC                  | Instrument/Valve/Rotary valve NC                      |
| 17            | Piping       | Fittings         | Spade blind                      | Piping/Fittings/Spade blind                          |
| 18            | Piping       | Fittings         | Spade close blind (flanged)      | Piping/Fittings/Spade close blind (flanged)          |
| 19            | Piping       | Fittings         | Spade open blind(flanged)        | Piping/Fittings/Spade open blind(flanged)            |
| 20            | Piping       | Fittings         | Right concentric reducer         | Piping/Fittings/Right concentric reducer             |
| 21            | Piping       | Fittings         | Flanged connection               | Piping/Fittings/Flanged connection                   |
| 22            | Piping       | Fittings         | Heating coil tubes               | Piping/Fittings/Heating coil tubes                   |
| 23            | Piping       | Fittings         | Jacketed pipe                    | Piping/Fittings/Jacketed pipe                         |
| 24            | Piping       | Fittings         | Mid arrow flow direction         | Piping/Fittings/Mid arrow flow direction              |
| 25            | Instrument   | Valve            | Circle valve                     | Instrument/Valve/Circle valve                         |
| 26            | Instrument   | Indicator        | Field mounted discrete indicator | Instrument/Indicator/Field mounted discrete indicator |
| 27            | Instrument   | Indicator        | Field mounted discrete recorder  | Instrument/Indicator/Field mounted discrete recorder  |
| 28            | Instrument   | Indicator        | discrete with Pri. Loc access    | Instrument/Indicator/discrete with Pri. Loc access    |
| 29            | Instrument   | Indicator        | discrete with Aux Loc access     | Instrument/Indicator/discrete with Aux Loc access     |
| 30            | Instrument   | Indicator        | Solenoid actuator                | Instrument/Indicator/Solenoid actuator                |
| 31            | Instrument   | Indicator        | shared with Pri. Loc access      | Instrument/Indicator/shared with Pri. Loc access      |
| 32            | Instrument   | Indicator        | shared control logic             | Instrument/Indicator/shared control logic             |
