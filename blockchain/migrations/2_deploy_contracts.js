const IncidentReportRegistry = artifacts.require("IncidentReportRegistry");

module.exports = function(deployer) {
  deployer.deploy(IncidentReportRegistry);
};