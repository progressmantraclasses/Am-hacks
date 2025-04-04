// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title IncidentReportRegistry
 * @dev Contract for registering and managing crime/incident reports on blockchain
 */
contract IncidentReportRegistry {
    // Role-based access control
    mapping(address => bool) public authorities;
    address public admin;
    
    // Report status options
    enum Status { Reported, UnderInvestigation, Resolved, Closed }
    
    // Incident Report structure
    struct IncidentReport {
        uint256 id;
        address reporter;
        string incidentHash; // IPFS hash of detailed report data
        uint256 timestamp;
        Status status;
        string location; // Encrypted or hashed location data
        string incidentType;
        bool isAnonymous;
    }
    
    // Storage
    mapping(uint256 => IncidentReport) public reports;
    uint256 public reportCount = 0;
    
    // Events
    event ReportFiled(uint256 indexed reportId, address indexed reporter, string incidentType);
    event StatusUpdated(uint256 indexed reportId, Status status);
    event AuthorityAdded(address authority);
    event AuthorityRemoved(address authority);
    
    // Modifiers
    modifier onlyAdmin() {
        require(msg.sender == admin, "Only admin can perform this action");
        _;
    }
    
    modifier onlyAuthority() {
        require(authorities[msg.sender] || msg.sender == admin, "Only authorities can perform this action");
        _;
    }
    
    // Constructor
    constructor() {
        admin = msg.sender;
        authorities[msg.sender] = true;
    }
    
    /**
     * @dev File a new incident report
     * @param _incidentHash IPFS hash of detailed report data
     * @param _location Location data (can be encrypted)
     * @param _incidentType Type of incident/crime
     * @param _isAnonymous Whether reporter wants to remain anonymous
     */
    function fileReport(
        string memory _incidentHash,
        string memory _location,
        string memory _incidentType,
        bool _isAnonymous
    ) public returns (uint256) {
        reportCount++;
        
        reports[reportCount] = IncidentReport({
            id: reportCount,
            reporter: _isAnonymous ? address(0) : msg.sender,
            incidentHash: _incidentHash,
            timestamp: block.timestamp,
            status: Status.Reported,
            location: _location,
            incidentType: _incidentType,
            isAnonymous: _isAnonymous
        });
        
        emit ReportFiled(reportCount, msg.sender, _incidentType);
        return reportCount;
    }
    
    /**
     * @dev Update the status of an incident report
     * @param _reportId ID of the report to update
     * @param _status New status to set
     */
    function updateReportStatus(uint256 _reportId, Status _status) public onlyAuthority {
        require(_reportId <= reportCount && _reportId > 0, "Report does not exist");
        
        IncidentReport storage report = reports[_reportId];
        report.status = _status;
        
        emit StatusUpdated(_reportId, _status);
    }
    
    /**
     * @dev Add a new authority
     * @param _authority Address of the new authority
     */
    function addAuthority(address _authority) public onlyAdmin {
        require(!authorities[_authority], "Address is already an authority");
        authorities[_authority] = true;
        emit AuthorityAdded(_authority);
    }
    
    /**
     * @dev Remove an authority
     * @param _authority Address of the authority to remove
     */
    function removeAuthority(address _authority) public onlyAdmin {
        require(authorities[_authority], "Address is not an authority");
        require(_authority != admin, "Cannot remove admin as authority");
        authorities[_authority] = false;
        emit AuthorityRemoved(_authority);
    }
    
    /**
     * @dev Get incident report details
     * @param _reportId ID of the report to retrieve
     */
    function getReport(uint256 _reportId) public view returns (
        uint256 id,
        address reporter,
        string memory incidentHash,
        uint256 timestamp,
        Status status,
        string memory location,
        string memory incidentType,
        bool isAnonymous
    ) {
        require(_reportId <= reportCount && _reportId > 0, "Report does not exist");
        IncidentReport storage report = reports[_reportId];
        
        return (
            report.id,
            report.reporter,
            report.incidentHash,
            report.timestamp,
            report.status,
            report.location,
            report.incidentType,
            report.isAnonymous
        );
    }
}