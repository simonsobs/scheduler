Calibration Targets 
======================


Notes on pointing parameters

* [For SATs](https://simonsobs.atlassian.net/wiki/spaces/PRO/pages/196509773/SATP+Encoder+Coordinates): roll = -boresight_enc
* [For the LAT](https://simonsobs.atlassian.net/wiki/spaces/PRO/pages/165380132/LAT+Encoder+Coordinates): roll = el_enc - 60 - corotator_enc


> The first pass of a lot of this code was origionally written for the SATs, so we will use `boresight = -1*(el_enc - 60 - corotator)` as a conversion