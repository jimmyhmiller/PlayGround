namespace cufxstandards.com {
    
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/AccessProfile.xsd")]
    public partial class AccessProfileList {
        
        private AccessProfile[] accessProfileField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("accessProfile")]
        public AccessProfile[] accessProfile {
            get {
                return this.accessProfileField;
            }
            set {
                this.accessProfileField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/AccessProfile.xsd")]
    public partial class AccessProfile {
        
        private string accessProfileIdField;
        
        private Actor actorField;
        
        private string languageField;
        
        private string localeField;
        
        private string characterEncodingField;
        
        private System.DateTime createAccessDateTimeField;
        
        private bool createAccessDateTimeFieldSpecified;
        
        private bool savedAccessProfileField;
        
        private bool savedAccessProfileFieldSpecified;
        
        private SoftwareClient softwareClientField;
        
        private Device deviceField;
        
        /// <remarks/>
        public string accessProfileId {
            get {
                return this.accessProfileIdField;
            }
            set {
                this.accessProfileIdField = value;
            }
        }
        
        /// <remarks/>
        public Actor actor {
            get {
                return this.actorField;
            }
            set {
                this.actorField = value;
            }
        }
        
        /// <remarks/>
        public string language {
            get {
                return this.languageField;
            }
            set {
                this.languageField = value;
            }
        }
        
        /// <remarks/>
        public string locale {
            get {
                return this.localeField;
            }
            set {
                this.localeField = value;
            }
        }
        
        /// <remarks/>
        public string characterEncoding {
            get {
                return this.characterEncodingField;
            }
            set {
                this.characterEncodingField = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime createAccessDateTime {
            get {
                return this.createAccessDateTimeField;
            }
            set {
                this.createAccessDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool createAccessDateTimeSpecified {
            get {
                return this.createAccessDateTimeFieldSpecified;
            }
            set {
                this.createAccessDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public bool savedAccessProfile {
            get {
                return this.savedAccessProfileField;
            }
            set {
                this.savedAccessProfileField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool savedAccessProfileSpecified {
            get {
                return this.savedAccessProfileFieldSpecified;
            }
            set {
                this.savedAccessProfileFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public SoftwareClient softwareClient {
            get {
                return this.softwareClientField;
            }
            set {
                this.softwareClientField = value;
            }
        }
        
        /// <remarks/>
        public Device device {
            get {
                return this.deviceField;
            }
            set {
                this.deviceField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/AccessProfile.xsd")]
    public partial class Actor {
        
        private string itemField;
        
        private ItemChoiceType itemElementNameField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("fiUserId", typeof(string))]
        [System.Xml.Serialization.XmlElementAttribute("partyId", typeof(string))]
        [System.Xml.Serialization.XmlElementAttribute("relationshipId", typeof(string))]
        [System.Xml.Serialization.XmlChoiceIdentifierAttribute("ItemElementName")]
        public string Item {
            get {
                return this.itemField;
            }
            set {
                this.itemField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public ItemChoiceType ItemElementName {
            get {
                return this.itemElementNameField;
            }
            set {
                this.itemElementNameField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/AccessProfile.xsd", IncludeInSchema=false)]
    public enum ItemChoiceType {
        
        /// <remarks/>
        fiUserId,
        
        /// <remarks/>
        partyId,
        
        /// <remarks/>
        relationshipId,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/AccessProfile.xsd")]
    public partial class Device {
        
        private string deviceIdField;
        
        private string deviceTypeField;
        
        private string deviceNameField;
        
        private string rawDeviceDetailField;
        
        private string ipAddressField;
        
        private string manufacturerField;
        
        /// <remarks/>
        public string deviceId {
            get {
                return this.deviceIdField;
            }
            set {
                this.deviceIdField = value;
            }
        }
        
        /// <remarks/>
        public string deviceType {
            get {
                return this.deviceTypeField;
            }
            set {
                this.deviceTypeField = value;
            }
        }
        
        /// <remarks/>
        public string deviceName {
            get {
                return this.deviceNameField;
            }
            set {
                this.deviceNameField = value;
            }
        }
        
        /// <remarks/>
        public string rawDeviceDetail {
            get {
                return this.rawDeviceDetailField;
            }
            set {
                this.rawDeviceDetailField = value;
            }
        }
        
        /// <remarks/>
        public string ipAddress {
            get {
                return this.ipAddressField;
            }
            set {
                this.ipAddressField = value;
            }
        }
        
        /// <remarks/>
        public string manufacturer {
            get {
                return this.manufacturerField;
            }
            set {
                this.manufacturerField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/AccessProfile.xsd")]
    public partial class SoftwareClient {
        
        private string softwareClientIdField;
        
        private string rawSoftwareClientDetailField;
        
        private string browserNameField;
        
        private string browserVersionField;
        
        private string operatingSystemNameField;
        
        private string operatingSystemVersionField;
        
        /// <remarks/>
        public string softwareClientId {
            get {
                return this.softwareClientIdField;
            }
            set {
                this.softwareClientIdField = value;
            }
        }
        
        /// <remarks/>
        public string rawSoftwareClientDetail {
            get {
                return this.rawSoftwareClientDetailField;
            }
            set {
                this.rawSoftwareClientDetailField = value;
            }
        }
        
        /// <remarks/>
        public string browserName {
            get {
                return this.browserNameField;
            }
            set {
                this.browserNameField = value;
            }
        }
        
        /// <remarks/>
        public string browserVersion {
            get {
                return this.browserVersionField;
            }
            set {
                this.browserVersionField = value;
            }
        }
        
        /// <remarks/>
        public string operatingSystemName {
            get {
                return this.operatingSystemNameField;
            }
            set {
                this.operatingSystemNameField = value;
            }
        }
        
        /// <remarks/>
        public string operatingSystemVersion {
            get {
                return this.operatingSystemVersionField;
            }
            set {
                this.operatingSystemVersionField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/AccessProfileFilter.xsd")]
    public partial class AccessProfileFilter {
        
        private string[] accessProfileIdListField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("accessProfileId", Namespace="http://cufxstandards.com/v3/AccessProfile.xsd", IsNullable=false)]
        public string[] accessProfileIdList {
            get {
                return this.accessProfileIdListField;
            }
            set {
                this.accessProfileIdListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/AccessProfileMessage.xsd")]
    public partial class AccessProfileMessage {
        
        private MessageContext messageContextField;
        
        private AccessProfileFilter accessProfileFilterField;
        
        private AccessProfile[] accessProfileListField;
        
        /// <remarks/>
        public MessageContext messageContext {
            get {
                return this.messageContextField;
            }
            set {
                this.messageContextField = value;
            }
        }
        
        /// <remarks/>
        public AccessProfileFilter accessProfileFilter {
            get {
                return this.accessProfileFilterField;
            }
            set {
                this.accessProfileFilterField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("accessProfile", Namespace="http://cufxstandards.com/v3/AccessProfile.xsd", IsNullable=false)]
        public AccessProfile[] accessProfileList {
            get {
                return this.accessProfileListField;
            }
            set {
                this.accessProfileListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/MessageContext.xsd")]
    public partial class MessageContext {
        
        private string requestIdField;
        
        private string vendorIdField;
        
        private string appIdField;
        
        private string fiIdField;
        
        private string dataSourceIdField;
        
        private Environment environmentField;
        
        private bool environmentFieldSpecified;
        
        private ReturnDataFilter returnDataFilterField;
        
        private bool includeBlankFieldsField;
        
        private bool includeZeroNumericsField;
        
        private User[] userField;
        
        private ValuePair[] customDataField;
        
        public MessageContext() {
            this.returnDataFilterField = ReturnDataFilter.All;
            this.includeBlankFieldsField = true;
            this.includeZeroNumericsField = true;
        }
        
        /// <remarks/>
        public string requestId {
            get {
                return this.requestIdField;
            }
            set {
                this.requestIdField = value;
            }
        }
        
        /// <remarks/>
        public string vendorId {
            get {
                return this.vendorIdField;
            }
            set {
                this.vendorIdField = value;
            }
        }
        
        /// <remarks/>
        public string appId {
            get {
                return this.appIdField;
            }
            set {
                this.appIdField = value;
            }
        }
        
        /// <remarks/>
        public string fiId {
            get {
                return this.fiIdField;
            }
            set {
                this.fiIdField = value;
            }
        }
        
        /// <remarks/>
        public string dataSourceId {
            get {
                return this.dataSourceIdField;
            }
            set {
                this.dataSourceIdField = value;
            }
        }
        
        /// <remarks/>
        public Environment environment {
            get {
                return this.environmentField;
            }
            set {
                this.environmentField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool environmentSpecified {
            get {
                return this.environmentFieldSpecified;
            }
            set {
                this.environmentFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.ComponentModel.DefaultValueAttribute(ReturnDataFilter.All)]
        public ReturnDataFilter returnDataFilter {
            get {
                return this.returnDataFilterField;
            }
            set {
                this.returnDataFilterField = value;
            }
        }
        
        /// <remarks/>
        public bool includeBlankFields {
            get {
                return this.includeBlankFieldsField;
            }
            set {
                this.includeBlankFieldsField = value;
            }
        }
        
        /// <remarks/>
        public bool includeZeroNumerics {
            get {
                return this.includeZeroNumericsField;
            }
            set {
                this.includeZeroNumericsField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("user")]
        public User[] user {
            get {
                return this.userField;
            }
            set {
                this.userField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("valuePair", Namespace="http://cufxstandards.com/v3/Common.xsd", IsNullable=false)]
        public ValuePair[] customData {
            get {
                return this.customDataField;
            }
            set {
                this.customDataField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/MessageContext.xsd")]
    public enum Environment {
        
        /// <remarks/>
        Development,
        
        /// <remarks/>
        UAT,
        
        /// <remarks/>
        Training,
        
        /// <remarks/>
        QA,
        
        /// <remarks/>
        Production,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/MessageContext.xsd")]
    public enum ReturnDataFilter {
        
        /// <remarks/>
        All,
        
        /// <remarks/>
        OnlyCreatedOrChangedData,
        
        /// <remarks/>
        None,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/MessageContext.xsd")]
    public partial class User {
        
        private string userIdField;
        
        private UserType userTypeField;
        
        private bool userTypeFieldSpecified;
        
        /// <remarks/>
        public string userId {
            get {
                return this.userIdField;
            }
            set {
                this.userIdField = value;
            }
        }
        
        /// <remarks/>
        public UserType userType {
            get {
                return this.userTypeField;
            }
            set {
                this.userTypeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool userTypeSpecified {
            get {
                return this.userTypeFieldSpecified;
            }
            set {
                this.userTypeFieldSpecified = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/MessageContext.xsd")]
    public enum UserType {
        
        /// <remarks/>
        EmployeeId,
        
        /// <remarks/>
        VendorEmployeeId,
        
        /// <remarks/>
        Anonymous,
        
        /// <remarks/>
        FIUserId,
        
        /// <remarks/>
        SystemAccountId,
        
        /// <remarks/>
        SecurityToken,
        
        /// <remarks/>
        Custom,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/SecureMessage.xsd")]
    public partial class SecureMessageUser : User {
        
        private string partyIdField;
        
        private string relationshipIdField;
        
        private string accountIdField;
        
        private string cardIdField;
        
        /// <remarks/>
        public string partyId {
            get {
                return this.partyIdField;
            }
            set {
                this.partyIdField = value;
            }
        }
        
        /// <remarks/>
        public string relationshipId {
            get {
                return this.relationshipIdField;
            }
            set {
                this.relationshipIdField = value;
            }
        }
        
        /// <remarks/>
        public string accountId {
            get {
                return this.accountIdField;
            }
            set {
                this.accountIdField = value;
            }
        }
        
        /// <remarks/>
        public string cardId {
            get {
                return this.cardIdField;
            }
            set {
                this.cardIdField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Common.xsd")]
    public partial class ValuePair {
        
        private string nameField;
        
        private string valueField;
        
        /// <remarks/>
        public string name {
            get {
                return this.nameField;
            }
            set {
                this.nameField = value;
            }
        }
        
        /// <remarks/>
        public string value {
            get {
                return this.valueField;
            }
            set {
                this.valueField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Account.xsd")]
    public partial class AccountList {
        
        private Account[] accountField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("account")]
        public Account[] account {
            get {
                return this.accountField;
            }
            set {
                this.accountField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Account.xsd")]
    public partial class Account {
        
        private string accountIdField;
        
        private IdType idTypeField;
        
        private bool idTypeFieldSpecified;
        
        private string descriptionField;
        
        private AccountType typeField;
        
        private bool typeFieldSpecified;
        
        private string subTypeField;
        
        private System.DateTime openDateField;
        
        private bool openDateFieldSpecified;
        
        private System.DateTime closeDateField;
        
        private bool closeDateFieldSpecified;
        
        private System.DateTime accountEscheatDateField;
        
        private bool accountEscheatDateFieldSpecified;
        
        private Money accountEscheatAmountField;
        
        private System.DateTime chargeOffDateField;
        
        private bool chargeOffDateFieldSpecified;
        
        private Money chargeOffAmountField;
        
        private string accountNickNameField;
        
        private string micrAccountNumberField;
        
        private string branchField;
        
        private Money actualBalanceField;
        
        private Money availableBalanceField;
        
        private Money minimumBalanceField;
        
        private string routingNumberField;
        
        private Address externalAccountBankAddressField;
        
        private string externalAccountSWIFTCodeField;
        
        private string externalAccountIBANCodeField;
        
        private string externalAccountBankCodeField;
        
        private bool externalAccountFlagField;
        
        private bool externalAccountFlagFieldSpecified;
        
        private bool externalAccountVerifiedField;
        
        private bool externalAccountVerifiedFieldSpecified;
        
        private bool externalTransferFromField;
        
        private bool externalTransferFromFieldSpecified;
        
        private bool externalTransferToField;
        
        private bool externalTransferToFieldSpecified;
        
        private bool transferFromField;
        
        private bool transferFromFieldSpecified;
        
        private bool transferToField;
        
        private bool transferToFieldSpecified;
        
        private RateType rateTypeField;
        
        private bool rateTypeFieldSpecified;
        
        private string sourceCodeField;
        
        private string[] partyIdListField;
        
        private string relationshipIdField;
        
        private Note[] accountNoteListField;
        
        private ValuePair[] customDataField;
        
        private Meta metaField;
        
        private TransactionListTransaction[] transactionListField;
        
        /// <remarks/>
        public string accountId {
            get {
                return this.accountIdField;
            }
            set {
                this.accountIdField = value;
            }
        }
        
        /// <remarks/>
        public IdType idType {
            get {
                return this.idTypeField;
            }
            set {
                this.idTypeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool idTypeSpecified {
            get {
                return this.idTypeFieldSpecified;
            }
            set {
                this.idTypeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string description {
            get {
                return this.descriptionField;
            }
            set {
                this.descriptionField = value;
            }
        }
        
        /// <remarks/>
        public AccountType type {
            get {
                return this.typeField;
            }
            set {
                this.typeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool typeSpecified {
            get {
                return this.typeFieldSpecified;
            }
            set {
                this.typeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string subType {
            get {
                return this.subTypeField;
            }
            set {
                this.subTypeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime openDate {
            get {
                return this.openDateField;
            }
            set {
                this.openDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool openDateSpecified {
            get {
                return this.openDateFieldSpecified;
            }
            set {
                this.openDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime closeDate {
            get {
                return this.closeDateField;
            }
            set {
                this.closeDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool closeDateSpecified {
            get {
                return this.closeDateFieldSpecified;
            }
            set {
                this.closeDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime accountEscheatDate {
            get {
                return this.accountEscheatDateField;
            }
            set {
                this.accountEscheatDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool accountEscheatDateSpecified {
            get {
                return this.accountEscheatDateFieldSpecified;
            }
            set {
                this.accountEscheatDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public Money accountEscheatAmount {
            get {
                return this.accountEscheatAmountField;
            }
            set {
                this.accountEscheatAmountField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime chargeOffDate {
            get {
                return this.chargeOffDateField;
            }
            set {
                this.chargeOffDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool chargeOffDateSpecified {
            get {
                return this.chargeOffDateFieldSpecified;
            }
            set {
                this.chargeOffDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public Money chargeOffAmount {
            get {
                return this.chargeOffAmountField;
            }
            set {
                this.chargeOffAmountField = value;
            }
        }
        
        /// <remarks/>
        public string accountNickName {
            get {
                return this.accountNickNameField;
            }
            set {
                this.accountNickNameField = value;
            }
        }
        
        /// <remarks/>
        public string micrAccountNumber {
            get {
                return this.micrAccountNumberField;
            }
            set {
                this.micrAccountNumberField = value;
            }
        }
        
        /// <remarks/>
        public string branch {
            get {
                return this.branchField;
            }
            set {
                this.branchField = value;
            }
        }
        
        /// <remarks/>
        public Money actualBalance {
            get {
                return this.actualBalanceField;
            }
            set {
                this.actualBalanceField = value;
            }
        }
        
        /// <remarks/>
        public Money availableBalance {
            get {
                return this.availableBalanceField;
            }
            set {
                this.availableBalanceField = value;
            }
        }
        
        /// <remarks/>
        public Money minimumBalance {
            get {
                return this.minimumBalanceField;
            }
            set {
                this.minimumBalanceField = value;
            }
        }
        
        /// <remarks/>
        public string routingNumber {
            get {
                return this.routingNumberField;
            }
            set {
                this.routingNumberField = value;
            }
        }
        
        /// <remarks/>
        public Address externalAccountBankAddress {
            get {
                return this.externalAccountBankAddressField;
            }
            set {
                this.externalAccountBankAddressField = value;
            }
        }
        
        /// <remarks/>
        public string externalAccountSWIFTCode {
            get {
                return this.externalAccountSWIFTCodeField;
            }
            set {
                this.externalAccountSWIFTCodeField = value;
            }
        }
        
        /// <remarks/>
        public string externalAccountIBANCode {
            get {
                return this.externalAccountIBANCodeField;
            }
            set {
                this.externalAccountIBANCodeField = value;
            }
        }
        
        /// <remarks/>
        public string externalAccountBankCode {
            get {
                return this.externalAccountBankCodeField;
            }
            set {
                this.externalAccountBankCodeField = value;
            }
        }
        
        /// <remarks/>
        public bool externalAccountFlag {
            get {
                return this.externalAccountFlagField;
            }
            set {
                this.externalAccountFlagField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool externalAccountFlagSpecified {
            get {
                return this.externalAccountFlagFieldSpecified;
            }
            set {
                this.externalAccountFlagFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public bool externalAccountVerified {
            get {
                return this.externalAccountVerifiedField;
            }
            set {
                this.externalAccountVerifiedField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool externalAccountVerifiedSpecified {
            get {
                return this.externalAccountVerifiedFieldSpecified;
            }
            set {
                this.externalAccountVerifiedFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public bool externalTransferFrom {
            get {
                return this.externalTransferFromField;
            }
            set {
                this.externalTransferFromField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool externalTransferFromSpecified {
            get {
                return this.externalTransferFromFieldSpecified;
            }
            set {
                this.externalTransferFromFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public bool externalTransferTo {
            get {
                return this.externalTransferToField;
            }
            set {
                this.externalTransferToField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool externalTransferToSpecified {
            get {
                return this.externalTransferToFieldSpecified;
            }
            set {
                this.externalTransferToFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public bool transferFrom {
            get {
                return this.transferFromField;
            }
            set {
                this.transferFromField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool transferFromSpecified {
            get {
                return this.transferFromFieldSpecified;
            }
            set {
                this.transferFromFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public bool transferTo {
            get {
                return this.transferToField;
            }
            set {
                this.transferToField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool transferToSpecified {
            get {
                return this.transferToFieldSpecified;
            }
            set {
                this.transferToFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public RateType rateType {
            get {
                return this.rateTypeField;
            }
            set {
                this.rateTypeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool rateTypeSpecified {
            get {
                return this.rateTypeFieldSpecified;
            }
            set {
                this.rateTypeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string sourceCode {
            get {
                return this.sourceCodeField;
            }
            set {
                this.sourceCodeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("partyId", Namespace="http://cufxstandards.com/v3/Party.xsd", IsNullable=false)]
        public string[] partyIdList {
            get {
                return this.partyIdListField;
            }
            set {
                this.partyIdListField = value;
            }
        }
        
        /// <remarks/>
        public string relationshipId {
            get {
                return this.relationshipIdField;
            }
            set {
                this.relationshipIdField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("note", Namespace="http://cufxstandards.com/v3/Common.xsd", IsNullable=false)]
        public Note[] accountNoteList {
            get {
                return this.accountNoteListField;
            }
            set {
                this.accountNoteListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("valuePair", Namespace="http://cufxstandards.com/v3/Common.xsd", IsNullable=false)]
        public ValuePair[] customData {
            get {
                return this.customDataField;
            }
            set {
                this.customDataField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(IsNullable=true)]
        public Meta meta {
            get {
                return this.metaField;
            }
            set {
                this.metaField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("transaction", Namespace="http://cufxstandards.com/v3/Transaction.xsd")]
        public TransactionListTransaction[] transactionList {
            get {
                return this.transactionListField;
            }
            set {
                this.transactionListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Account.xsd")]
    public enum IdType {
        
        /// <remarks/>
        Reserved,
        
        /// <remarks/>
        Actual,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Account.xsd")]
    public enum AccountType {
        
        /// <remarks/>
        Checking,
        
        /// <remarks/>
        Savings,
        
        /// <remarks/>
        Loan,
        
        /// <remarks/>
        CreditCard,
        
        /// <remarks/>
        LineOfCredit,
        
        /// <remarks/>
        Mortgage,
        
        /// <remarks/>
        Investment,
        
        /// <remarks/>
        PrePaidCard,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Common.xsd")]
    public partial class Money {
        
        private decimal valueField;
        
        private ISOCurrencyCodeType currencyCodeField;
        
        private decimal exchangeRateField;
        
        private bool exchangeRateFieldSpecified;
        
        public Money() {
            this.currencyCodeField = ISOCurrencyCodeType.USD;
        }
        
        /// <remarks/>
        public decimal value {
            get {
                return this.valueField;
            }
            set {
                this.valueField = value;
            }
        }
        
        /// <remarks/>
        public ISOCurrencyCodeType currencyCode {
            get {
                return this.currencyCodeField;
            }
            set {
                this.currencyCodeField = value;
            }
        }
        
        /// <remarks/>
        public decimal exchangeRate {
            get {
                return this.exchangeRateField;
            }
            set {
                this.exchangeRateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool exchangeRateSpecified {
            get {
                return this.exchangeRateFieldSpecified;
            }
            set {
                this.exchangeRateFieldSpecified = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/ISOCurrencyCodeType.xsd")]
    public enum ISOCurrencyCodeType {
        
        /// <remarks/>
        AED,
        
        /// <remarks/>
        AFN,
        
        /// <remarks/>
        ALL,
        
        /// <remarks/>
        AMD,
        
        /// <remarks/>
        ANG,
        
        /// <remarks/>
        AOA,
        
        /// <remarks/>
        ARS,
        
        /// <remarks/>
        AUD,
        
        /// <remarks/>
        AWG,
        
        /// <remarks/>
        AZN,
        
        /// <remarks/>
        BAM,
        
        /// <remarks/>
        BBD,
        
        /// <remarks/>
        BDT,
        
        /// <remarks/>
        BGN,
        
        /// <remarks/>
        BHD,
        
        /// <remarks/>
        BIF,
        
        /// <remarks/>
        BMD,
        
        /// <remarks/>
        BND,
        
        /// <remarks/>
        BOB,
        
        /// <remarks/>
        BRL,
        
        /// <remarks/>
        BSD,
        
        /// <remarks/>
        BTN,
        
        /// <remarks/>
        BWP,
        
        /// <remarks/>
        BYR,
        
        /// <remarks/>
        BZD,
        
        /// <remarks/>
        CAD,
        
        /// <remarks/>
        CDF,
        
        /// <remarks/>
        CHF,
        
        /// <remarks/>
        CLP,
        
        /// <remarks/>
        CNY,
        
        /// <remarks/>
        COP,
        
        /// <remarks/>
        CRC,
        
        /// <remarks/>
        CUP,
        
        /// <remarks/>
        CVE,
        
        /// <remarks/>
        CZK,
        
        /// <remarks/>
        DJF,
        
        /// <remarks/>
        DKK,
        
        /// <remarks/>
        DOP,
        
        /// <remarks/>
        DZD,
        
        /// <remarks/>
        EEK,
        
        /// <remarks/>
        EGP,
        
        /// <remarks/>
        ERN,
        
        /// <remarks/>
        ETB,
        
        /// <remarks/>
        EUR,
        
        /// <remarks/>
        FJD,
        
        /// <remarks/>
        FKP,
        
        /// <remarks/>
        GBP,
        
        /// <remarks/>
        GEL,
        
        /// <remarks/>
        GHS,
        
        /// <remarks/>
        GIP,
        
        /// <remarks/>
        GMD,
        
        /// <remarks/>
        GNF,
        
        /// <remarks/>
        GTQ,
        
        /// <remarks/>
        GYD,
        
        /// <remarks/>
        GWP,
        
        /// <remarks/>
        HKD,
        
        /// <remarks/>
        HNL,
        
        /// <remarks/>
        HRK,
        
        /// <remarks/>
        HTG,
        
        /// <remarks/>
        HUF,
        
        /// <remarks/>
        IDR,
        
        /// <remarks/>
        ILS,
        
        /// <remarks/>
        INR,
        
        /// <remarks/>
        IQD,
        
        /// <remarks/>
        IRR,
        
        /// <remarks/>
        ISK,
        
        /// <remarks/>
        JMD,
        
        /// <remarks/>
        JOD,
        
        /// <remarks/>
        JPY,
        
        /// <remarks/>
        KES,
        
        /// <remarks/>
        KGS,
        
        /// <remarks/>
        KHR,
        
        /// <remarks/>
        KMF,
        
        /// <remarks/>
        KPW,
        
        /// <remarks/>
        KRW,
        
        /// <remarks/>
        KWD,
        
        /// <remarks/>
        KYD,
        
        /// <remarks/>
        KZT,
        
        /// <remarks/>
        LAK,
        
        /// <remarks/>
        LBP,
        
        /// <remarks/>
        LKR,
        
        /// <remarks/>
        LRD,
        
        /// <remarks/>
        LSL,
        
        /// <remarks/>
        LTL,
        
        /// <remarks/>
        LVL,
        
        /// <remarks/>
        LYD,
        
        /// <remarks/>
        MAD,
        
        /// <remarks/>
        MDL,
        
        /// <remarks/>
        MGA,
        
        /// <remarks/>
        MKD,
        
        /// <remarks/>
        MMK,
        
        /// <remarks/>
        MNT,
        
        /// <remarks/>
        MOP,
        
        /// <remarks/>
        MRO,
        
        /// <remarks/>
        MUR,
        
        /// <remarks/>
        MVR,
        
        /// <remarks/>
        MWK,
        
        /// <remarks/>
        MXN,
        
        /// <remarks/>
        MYR,
        
        /// <remarks/>
        MZN,
        
        /// <remarks/>
        NAD,
        
        /// <remarks/>
        NGN,
        
        /// <remarks/>
        NIO,
        
        /// <remarks/>
        NOK,
        
        /// <remarks/>
        NPR,
        
        /// <remarks/>
        NZD,
        
        /// <remarks/>
        OMR,
        
        /// <remarks/>
        PAB,
        
        /// <remarks/>
        PEN,
        
        /// <remarks/>
        PGK,
        
        /// <remarks/>
        PHP,
        
        /// <remarks/>
        PKR,
        
        /// <remarks/>
        PLN,
        
        /// <remarks/>
        PYG,
        
        /// <remarks/>
        QAR,
        
        /// <remarks/>
        RON,
        
        /// <remarks/>
        RSD,
        
        /// <remarks/>
        RUB,
        
        /// <remarks/>
        RWF,
        
        /// <remarks/>
        SAR,
        
        /// <remarks/>
        SBD,
        
        /// <remarks/>
        SCR,
        
        /// <remarks/>
        SDG,
        
        /// <remarks/>
        SEK,
        
        /// <remarks/>
        SGD,
        
        /// <remarks/>
        SHP,
        
        /// <remarks/>
        SKK,
        
        /// <remarks/>
        SLL,
        
        /// <remarks/>
        SOS,
        
        /// <remarks/>
        SRD,
        
        /// <remarks/>
        STD,
        
        /// <remarks/>
        SVC,
        
        /// <remarks/>
        SYP,
        
        /// <remarks/>
        SZL,
        
        /// <remarks/>
        THB,
        
        /// <remarks/>
        TJS,
        
        /// <remarks/>
        TMM,
        
        /// <remarks/>
        TND,
        
        /// <remarks/>
        TOP,
        
        /// <remarks/>
        TRY,
        
        /// <remarks/>
        TTD,
        
        /// <remarks/>
        TWD,
        
        /// <remarks/>
        TZS,
        
        /// <remarks/>
        UAH,
        
        /// <remarks/>
        UGX,
        
        /// <remarks/>
        USD,
        
        /// <remarks/>
        UYU,
        
        /// <remarks/>
        UZS,
        
        /// <remarks/>
        VEF,
        
        /// <remarks/>
        VND,
        
        /// <remarks/>
        VUV,
        
        /// <remarks/>
        WST,
        
        /// <remarks/>
        XAF,
        
        /// <remarks/>
        XAG,
        
        /// <remarks/>
        XAU,
        
        /// <remarks/>
        XCD,
        
        /// <remarks/>
        XDR,
        
        /// <remarks/>
        XOF,
        
        /// <remarks/>
        XPD,
        
        /// <remarks/>
        XPF,
        
        /// <remarks/>
        XPT,
        
        /// <remarks/>
        YER,
        
        /// <remarks/>
        ZAR,
        
        /// <remarks/>
        ZMK,
        
        /// <remarks/>
        ZWR,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Common.xsd")]
    public partial class Address {
        
        private string line1Field;
        
        private string line2Field;
        
        private string line3Field;
        
        private string cityField;
        
        private string stateProvinceField;
        
        private string postalCodeField;
        
        private ISOCountryCodeType countryCodeField;
        
        private bool countryCodeFieldSpecified;
        
        private string countryField;
        
        private Coordinate coordinateField;
        
        /// <remarks/>
        public string line1 {
            get {
                return this.line1Field;
            }
            set {
                this.line1Field = value;
            }
        }
        
        /// <remarks/>
        public string line2 {
            get {
                return this.line2Field;
            }
            set {
                this.line2Field = value;
            }
        }
        
        /// <remarks/>
        public string line3 {
            get {
                return this.line3Field;
            }
            set {
                this.line3Field = value;
            }
        }
        
        /// <remarks/>
        public string city {
            get {
                return this.cityField;
            }
            set {
                this.cityField = value;
            }
        }
        
        /// <remarks/>
        public string stateProvince {
            get {
                return this.stateProvinceField;
            }
            set {
                this.stateProvinceField = value;
            }
        }
        
        /// <remarks/>
        public string postalCode {
            get {
                return this.postalCodeField;
            }
            set {
                this.postalCodeField = value;
            }
        }
        
        /// <remarks/>
        public ISOCountryCodeType countryCode {
            get {
                return this.countryCodeField;
            }
            set {
                this.countryCodeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool countryCodeSpecified {
            get {
                return this.countryCodeFieldSpecified;
            }
            set {
                this.countryCodeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string country {
            get {
                return this.countryField;
            }
            set {
                this.countryField = value;
            }
        }
        
        /// <remarks/>
        public Coordinate coordinate {
            get {
                return this.coordinateField;
            }
            set {
                this.coordinateField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/ISOCountryCodeType-V2006.xsd")]
    public enum ISOCountryCodeType {
        
        /// <remarks/>
        AD,
        
        /// <remarks/>
        AE,
        
        /// <remarks/>
        AF,
        
        /// <remarks/>
        AG,
        
        /// <remarks/>
        AI,
        
        /// <remarks/>
        AL,
        
        /// <remarks/>
        AM,
        
        /// <remarks/>
        AN,
        
        /// <remarks/>
        AO,
        
        /// <remarks/>
        AQ,
        
        /// <remarks/>
        AR,
        
        /// <remarks/>
        AS,
        
        /// <remarks/>
        AT,
        
        /// <remarks/>
        AU,
        
        /// <remarks/>
        AW,
        
        /// <remarks/>
        AX,
        
        /// <remarks/>
        AZ,
        
        /// <remarks/>
        BA,
        
        /// <remarks/>
        BB,
        
        /// <remarks/>
        BD,
        
        /// <remarks/>
        BE,
        
        /// <remarks/>
        BF,
        
        /// <remarks/>
        BG,
        
        /// <remarks/>
        BH,
        
        /// <remarks/>
        BI,
        
        /// <remarks/>
        BJ,
        
        /// <remarks/>
        BL,
        
        /// <remarks/>
        BM,
        
        /// <remarks/>
        BN,
        
        /// <remarks/>
        BO,
        
        /// <remarks/>
        BR,
        
        /// <remarks/>
        BS,
        
        /// <remarks/>
        BT,
        
        /// <remarks/>
        BV,
        
        /// <remarks/>
        BW,
        
        /// <remarks/>
        BY,
        
        /// <remarks/>
        BZ,
        
        /// <remarks/>
        CA,
        
        /// <remarks/>
        CC,
        
        /// <remarks/>
        CD,
        
        /// <remarks/>
        CF,
        
        /// <remarks/>
        CG,
        
        /// <remarks/>
        CH,
        
        /// <remarks/>
        CI,
        
        /// <remarks/>
        CK,
        
        /// <remarks/>
        CL,
        
        /// <remarks/>
        CM,
        
        /// <remarks/>
        CN,
        
        /// <remarks/>
        CO,
        
        /// <remarks/>
        CR,
        
        /// <remarks/>
        CU,
        
        /// <remarks/>
        CV,
        
        /// <remarks/>
        CX,
        
        /// <remarks/>
        CY,
        
        /// <remarks/>
        CZ,
        
        /// <remarks/>
        DE,
        
        /// <remarks/>
        DJ,
        
        /// <remarks/>
        DK,
        
        /// <remarks/>
        DM,
        
        /// <remarks/>
        DO,
        
        /// <remarks/>
        DZ,
        
        /// <remarks/>
        EC,
        
        /// <remarks/>
        EE,
        
        /// <remarks/>
        EG,
        
        /// <remarks/>
        EH,
        
        /// <remarks/>
        ER,
        
        /// <remarks/>
        ES,
        
        /// <remarks/>
        ET,
        
        /// <remarks/>
        FI,
        
        /// <remarks/>
        FJ,
        
        /// <remarks/>
        FK,
        
        /// <remarks/>
        FM,
        
        /// <remarks/>
        FO,
        
        /// <remarks/>
        FR,
        
        /// <remarks/>
        GA,
        
        /// <remarks/>
        GB,
        
        /// <remarks/>
        GD,
        
        /// <remarks/>
        GE,
        
        /// <remarks/>
        GF,
        
        /// <remarks/>
        GG,
        
        /// <remarks/>
        GH,
        
        /// <remarks/>
        GI,
        
        /// <remarks/>
        GL,
        
        /// <remarks/>
        GM,
        
        /// <remarks/>
        GN,
        
        /// <remarks/>
        GP,
        
        /// <remarks/>
        GQ,
        
        /// <remarks/>
        GR,
        
        /// <remarks/>
        GS,
        
        /// <remarks/>
        GT,
        
        /// <remarks/>
        GU,
        
        /// <remarks/>
        GW,
        
        /// <remarks/>
        GY,
        
        /// <remarks/>
        HK,
        
        /// <remarks/>
        HM,
        
        /// <remarks/>
        HN,
        
        /// <remarks/>
        HR,
        
        /// <remarks/>
        HT,
        
        /// <remarks/>
        HU,
        
        /// <remarks/>
        ID,
        
        /// <remarks/>
        IE,
        
        /// <remarks/>
        IL,
        
        /// <remarks/>
        IM,
        
        /// <remarks/>
        IN,
        
        /// <remarks/>
        IO,
        
        /// <remarks/>
        IQ,
        
        /// <remarks/>
        IR,
        
        /// <remarks/>
        IS,
        
        /// <remarks/>
        IT,
        
        /// <remarks/>
        JE,
        
        /// <remarks/>
        JM,
        
        /// <remarks/>
        JO,
        
        /// <remarks/>
        JP,
        
        /// <remarks/>
        KE,
        
        /// <remarks/>
        KG,
        
        /// <remarks/>
        KH,
        
        /// <remarks/>
        KI,
        
        /// <remarks/>
        KM,
        
        /// <remarks/>
        KN,
        
        /// <remarks/>
        KP,
        
        /// <remarks/>
        KR,
        
        /// <remarks/>
        KW,
        
        /// <remarks/>
        KY,
        
        /// <remarks/>
        KZ,
        
        /// <remarks/>
        LA,
        
        /// <remarks/>
        LB,
        
        /// <remarks/>
        LC,
        
        /// <remarks/>
        LI,
        
        /// <remarks/>
        LK,
        
        /// <remarks/>
        LR,
        
        /// <remarks/>
        LS,
        
        /// <remarks/>
        LT,
        
        /// <remarks/>
        LU,
        
        /// <remarks/>
        LV,
        
        /// <remarks/>
        LY,
        
        /// <remarks/>
        MA,
        
        /// <remarks/>
        MC,
        
        /// <remarks/>
        MD,
        
        /// <remarks/>
        ME,
        
        /// <remarks/>
        MF,
        
        /// <remarks/>
        MG,
        
        /// <remarks/>
        MH,
        
        /// <remarks/>
        MK,
        
        /// <remarks/>
        ML,
        
        /// <remarks/>
        MM,
        
        /// <remarks/>
        MN,
        
        /// <remarks/>
        MO,
        
        /// <remarks/>
        MP,
        
        /// <remarks/>
        MQ,
        
        /// <remarks/>
        MR,
        
        /// <remarks/>
        MS,
        
        /// <remarks/>
        MT,
        
        /// <remarks/>
        MU,
        
        /// <remarks/>
        MV,
        
        /// <remarks/>
        MW,
        
        /// <remarks/>
        MX,
        
        /// <remarks/>
        MY,
        
        /// <remarks/>
        MZ,
        
        /// <remarks/>
        NA,
        
        /// <remarks/>
        NC,
        
        /// <remarks/>
        NE,
        
        /// <remarks/>
        NF,
        
        /// <remarks/>
        NG,
        
        /// <remarks/>
        NI,
        
        /// <remarks/>
        NL,
        
        /// <remarks/>
        NO,
        
        /// <remarks/>
        NP,
        
        /// <remarks/>
        NR,
        
        /// <remarks/>
        NU,
        
        /// <remarks/>
        NZ,
        
        /// <remarks/>
        OM,
        
        /// <remarks/>
        PA,
        
        /// <remarks/>
        PE,
        
        /// <remarks/>
        PF,
        
        /// <remarks/>
        PG,
        
        /// <remarks/>
        PH,
        
        /// <remarks/>
        PK,
        
        /// <remarks/>
        PL,
        
        /// <remarks/>
        PM,
        
        /// <remarks/>
        PN,
        
        /// <remarks/>
        PR,
        
        /// <remarks/>
        PS,
        
        /// <remarks/>
        PT,
        
        /// <remarks/>
        PW,
        
        /// <remarks/>
        PY,
        
        /// <remarks/>
        QA,
        
        /// <remarks/>
        RE,
        
        /// <remarks/>
        RO,
        
        /// <remarks/>
        RS,
        
        /// <remarks/>
        RU,
        
        /// <remarks/>
        RW,
        
        /// <remarks/>
        SA,
        
        /// <remarks/>
        SB,
        
        /// <remarks/>
        SC,
        
        /// <remarks/>
        SD,
        
        /// <remarks/>
        SE,
        
        /// <remarks/>
        SG,
        
        /// <remarks/>
        SH,
        
        /// <remarks/>
        SI,
        
        /// <remarks/>
        SJ,
        
        /// <remarks/>
        SK,
        
        /// <remarks/>
        SL,
        
        /// <remarks/>
        SM,
        
        /// <remarks/>
        SN,
        
        /// <remarks/>
        SO,
        
        /// <remarks/>
        SR,
        
        /// <remarks/>
        ST,
        
        /// <remarks/>
        SV,
        
        /// <remarks/>
        SY,
        
        /// <remarks/>
        SZ,
        
        /// <remarks/>
        TC,
        
        /// <remarks/>
        TD,
        
        /// <remarks/>
        TF,
        
        /// <remarks/>
        TG,
        
        /// <remarks/>
        TH,
        
        /// <remarks/>
        TJ,
        
        /// <remarks/>
        TK,
        
        /// <remarks/>
        TL,
        
        /// <remarks/>
        TM,
        
        /// <remarks/>
        TN,
        
        /// <remarks/>
        TO,
        
        /// <remarks/>
        TR,
        
        /// <remarks/>
        TT,
        
        /// <remarks/>
        TV,
        
        /// <remarks/>
        TW,
        
        /// <remarks/>
        TZ,
        
        /// <remarks/>
        UA,
        
        /// <remarks/>
        UG,
        
        /// <remarks/>
        UM,
        
        /// <remarks/>
        US,
        
        /// <remarks/>
        UY,
        
        /// <remarks/>
        UZ,
        
        /// <remarks/>
        VA,
        
        /// <remarks/>
        VC,
        
        /// <remarks/>
        VE,
        
        /// <remarks/>
        VG,
        
        /// <remarks/>
        VI,
        
        /// <remarks/>
        VN,
        
        /// <remarks/>
        VU,
        
        /// <remarks/>
        WF,
        
        /// <remarks/>
        WS,
        
        /// <remarks/>
        YE,
        
        /// <remarks/>
        YT,
        
        /// <remarks/>
        ZA,
        
        /// <remarks/>
        ZM,
        
        /// <remarks/>
        ZW,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Common.xsd")]
    public partial class Coordinate {
        
        private decimal latitudeField;
        
        private decimal longitudeField;
        
        /// <remarks/>
        public decimal latitude {
            get {
                return this.latitudeField;
            }
            set {
                this.latitudeField = value;
            }
        }
        
        /// <remarks/>
        public decimal longitude {
            get {
                return this.longitudeField;
            }
            set {
                this.longitudeField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Contact.xsd")]
    public partial class ContactAddress : Address {
        
        private ContactAddressType typeField;
        
        private bool typeFieldSpecified;
        
        private System.DateTime effectiveDateTimeField;
        
        private bool effectiveDateTimeFieldSpecified;
        
        private System.DateTime expirationDateTimeField;
        
        private bool expirationDateTimeFieldSpecified;
        
        private string timeAtAddressField;
        
        private System.DateTime addressVerifyDateTimeField;
        
        private bool addressVerifyDateTimeFieldSpecified;
        
        /// <remarks/>
        public ContactAddressType type {
            get {
                return this.typeField;
            }
            set {
                this.typeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool typeSpecified {
            get {
                return this.typeFieldSpecified;
            }
            set {
                this.typeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime effectiveDateTime {
            get {
                return this.effectiveDateTimeField;
            }
            set {
                this.effectiveDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool effectiveDateTimeSpecified {
            get {
                return this.effectiveDateTimeFieldSpecified;
            }
            set {
                this.effectiveDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime expirationDateTime {
            get {
                return this.expirationDateTimeField;
            }
            set {
                this.expirationDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool expirationDateTimeSpecified {
            get {
                return this.expirationDateTimeFieldSpecified;
            }
            set {
                this.expirationDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="duration")]
        public string timeAtAddress {
            get {
                return this.timeAtAddressField;
            }
            set {
                this.timeAtAddressField = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime addressVerifyDateTime {
            get {
                return this.addressVerifyDateTimeField;
            }
            set {
                this.addressVerifyDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool addressVerifyDateTimeSpecified {
            get {
                return this.addressVerifyDateTimeFieldSpecified;
            }
            set {
                this.addressVerifyDateTimeFieldSpecified = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(AnonymousType=true, Namespace="http://cufxstandards.com/v3/Contact.xsd")]
    public enum ContactAddressType {
        
        /// <remarks/>
        Home,
        
        /// <remarks/>
        Work,
        
        /// <remarks/>
        Mailing,
        
        /// <remarks/>
        Previous,
        
        /// <remarks/>
        Temporary,
        
        /// <remarks/>
        CTR,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Account.xsd")]
    public enum RateType {
        
        /// <remarks/>
        Fixed,
        
        /// <remarks/>
        Variable,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Common.xsd")]
    public partial class Note {
        
        private CoreRecordType recordTypeField;
        
        private bool recordTypeFieldSpecified;
        
        private string noteCodeField;
        
        private string[] noteTextField;
        
        private System.DateTime noteCreatedDateTimeField;
        
        private bool noteCreatedDateTimeFieldSpecified;
        
        private System.DateTime noteExpirationDateField;
        
        private bool noteExpirationDateFieldSpecified;
        
        private string noteCreatorField;
        
        /// <remarks/>
        public CoreRecordType recordType {
            get {
                return this.recordTypeField;
            }
            set {
                this.recordTypeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool recordTypeSpecified {
            get {
                return this.recordTypeFieldSpecified;
            }
            set {
                this.recordTypeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string noteCode {
            get {
                return this.noteCodeField;
            }
            set {
                this.noteCodeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("noteText")]
        public string[] noteText {
            get {
                return this.noteTextField;
            }
            set {
                this.noteTextField = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime noteCreatedDateTime {
            get {
                return this.noteCreatedDateTimeField;
            }
            set {
                this.noteCreatedDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool noteCreatedDateTimeSpecified {
            get {
                return this.noteCreatedDateTimeFieldSpecified;
            }
            set {
                this.noteCreatedDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime noteExpirationDate {
            get {
                return this.noteExpirationDateField;
            }
            set {
                this.noteExpirationDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool noteExpirationDateSpecified {
            get {
                return this.noteExpirationDateFieldSpecified;
            }
            set {
                this.noteExpirationDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string noteCreator {
            get {
                return this.noteCreatorField;
            }
            set {
                this.noteCreatorField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Common.xsd")]
    public enum CoreRecordType {
        
        /// <remarks/>
        Account,
        
        /// <remarks/>
        Application,
        
        /// <remarks/>
        Loan,
        
        /// <remarks/>
        Portfolio,
        
        /// <remarks/>
        Deposit,
        
        /// <remarks/>
        ProductServiceRequest,
        
        /// <remarks/>
        Request,
        
        /// <remarks/>
        Activity,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Meta.xsd")]
    public partial class Meta {
        
        private object itemField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("checkingSavingsMeta", typeof(CheckingSavingsMeta))]
        [System.Xml.Serialization.XmlElementAttribute("creditCardMeta", typeof(CreditCardMeta))]
        [System.Xml.Serialization.XmlElementAttribute("investmentMeta", typeof(InvestmentMeta))]
        [System.Xml.Serialization.XmlElementAttribute("lineOfCreditMeta", typeof(LineOfCreditMeta))]
        [System.Xml.Serialization.XmlElementAttribute("loanMeta", typeof(LoanMeta))]
        [System.Xml.Serialization.XmlElementAttribute("mortgageMeta", typeof(MortgageMeta))]
        public object Item {
            get {
                return this.itemField;
            }
            set {
                this.itemField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Meta.xsd")]
    public partial class CheckingSavingsMeta {
        
        private decimal apyField;
        
        private bool apyFieldSpecified;
        
        /// <remarks/>
        public decimal apy {
            get {
                return this.apyField;
            }
            set {
                this.apyField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool apySpecified {
            get {
                return this.apyFieldSpecified;
            }
            set {
                this.apyFieldSpecified = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Meta.xsd")]
    public partial class CreditCardMeta {
        
        private string brandField;
        
        private Money creditLimitField;
        
        private Money minimumPaymentField;
        
        private System.DateTime currentDueDateField;
        
        private bool currentDueDateFieldSpecified;
        
        /// <remarks/>
        public string brand {
            get {
                return this.brandField;
            }
            set {
                this.brandField = value;
            }
        }
        
        /// <remarks/>
        public Money creditLimit {
            get {
                return this.creditLimitField;
            }
            set {
                this.creditLimitField = value;
            }
        }
        
        /// <remarks/>
        public Money minimumPayment {
            get {
                return this.minimumPaymentField;
            }
            set {
                this.minimumPaymentField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime currentDueDate {
            get {
                return this.currentDueDateField;
            }
            set {
                this.currentDueDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool currentDueDateSpecified {
            get {
                return this.currentDueDateFieldSpecified;
            }
            set {
                this.currentDueDateFieldSpecified = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Meta.xsd")]
    public partial class InvestmentMeta {
        
        private Money startingBalanceField;
        
        private System.DateTime maturityDateField;
        
        private bool maturityDateFieldSpecified;
        
        private decimal interestRateField;
        
        private bool interestRateFieldSpecified;
        
        private string compoundingFrequencyField;
        
        /// <remarks/>
        public Money startingBalance {
            get {
                return this.startingBalanceField;
            }
            set {
                this.startingBalanceField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime maturityDate {
            get {
                return this.maturityDateField;
            }
            set {
                this.maturityDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool maturityDateSpecified {
            get {
                return this.maturityDateFieldSpecified;
            }
            set {
                this.maturityDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public decimal interestRate {
            get {
                return this.interestRateField;
            }
            set {
                this.interestRateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool interestRateSpecified {
            get {
                return this.interestRateFieldSpecified;
            }
            set {
                this.interestRateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string compoundingFrequency {
            get {
                return this.compoundingFrequencyField;
            }
            set {
                this.compoundingFrequencyField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Meta.xsd")]
    public partial class LineOfCreditMeta {
        
        private decimal interestRateField;
        
        private bool interestRateFieldSpecified;
        
        private Money originalBalanceField;
        
        private Money creditLimitField;
        
        private Money minimumPaymentField;
        
        private System.DateTime currentDueDateField;
        
        private bool currentDueDateFieldSpecified;
        
        private Money currentPayoffBalanceField;
        
        /// <remarks/>
        public decimal interestRate {
            get {
                return this.interestRateField;
            }
            set {
                this.interestRateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool interestRateSpecified {
            get {
                return this.interestRateFieldSpecified;
            }
            set {
                this.interestRateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public Money originalBalance {
            get {
                return this.originalBalanceField;
            }
            set {
                this.originalBalanceField = value;
            }
        }
        
        /// <remarks/>
        public Money creditLimit {
            get {
                return this.creditLimitField;
            }
            set {
                this.creditLimitField = value;
            }
        }
        
        /// <remarks/>
        public Money minimumPayment {
            get {
                return this.minimumPaymentField;
            }
            set {
                this.minimumPaymentField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime currentDueDate {
            get {
                return this.currentDueDateField;
            }
            set {
                this.currentDueDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool currentDueDateSpecified {
            get {
                return this.currentDueDateFieldSpecified;
            }
            set {
                this.currentDueDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public Money currentPayoffBalance {
            get {
                return this.currentPayoffBalanceField;
            }
            set {
                this.currentPayoffBalanceField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Meta.xsd")]
    public partial class LoanMeta {
        
        private decimal interestRateField;
        
        private bool interestRateFieldSpecified;
        
        private Money creditLimitField;
        
        private decimal annualPercentageRateField;
        
        private bool annualPercentageRateFieldSpecified;
        
        private decimal rateField;
        
        private bool rateFieldSpecified;
        
        private Money minimumPaymentField;
        
        private System.DateTime maturityDateField;
        
        private bool maturityDateFieldSpecified;
        
        private Money originalBalanceField;
        
        private System.DateTime currentDueDateField;
        
        private bool currentDueDateFieldSpecified;
        
        private Money currentPayoffBalanceField;
        
        /// <remarks/>
        public decimal interestRate {
            get {
                return this.interestRateField;
            }
            set {
                this.interestRateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool interestRateSpecified {
            get {
                return this.interestRateFieldSpecified;
            }
            set {
                this.interestRateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public Money creditLimit {
            get {
                return this.creditLimitField;
            }
            set {
                this.creditLimitField = value;
            }
        }
        
        /// <remarks/>
        public decimal annualPercentageRate {
            get {
                return this.annualPercentageRateField;
            }
            set {
                this.annualPercentageRateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool annualPercentageRateSpecified {
            get {
                return this.annualPercentageRateFieldSpecified;
            }
            set {
                this.annualPercentageRateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public decimal rate {
            get {
                return this.rateField;
            }
            set {
                this.rateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool rateSpecified {
            get {
                return this.rateFieldSpecified;
            }
            set {
                this.rateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public Money minimumPayment {
            get {
                return this.minimumPaymentField;
            }
            set {
                this.minimumPaymentField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime maturityDate {
            get {
                return this.maturityDateField;
            }
            set {
                this.maturityDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool maturityDateSpecified {
            get {
                return this.maturityDateFieldSpecified;
            }
            set {
                this.maturityDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public Money originalBalance {
            get {
                return this.originalBalanceField;
            }
            set {
                this.originalBalanceField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime currentDueDate {
            get {
                return this.currentDueDateField;
            }
            set {
                this.currentDueDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool currentDueDateSpecified {
            get {
                return this.currentDueDateFieldSpecified;
            }
            set {
                this.currentDueDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public Money currentPayoffBalance {
            get {
                return this.currentPayoffBalanceField;
            }
            set {
                this.currentPayoffBalanceField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Meta.xsd")]
    public partial class MortgageMeta {
        
        private decimal interestRateField;
        
        private bool interestRateFieldSpecified;
        
        private Money originalBalanceField;
        
        private System.DateTime maturityDateField;
        
        private bool maturityDateFieldSpecified;
        
        private Money escrowBalanceField;
        
        private Money minimumPaymentField;
        
        private System.DateTime currentDueDateField;
        
        private bool currentDueDateFieldSpecified;
        
        private Money currentPayoffBalanceField;
        
        /// <remarks/>
        public decimal interestRate {
            get {
                return this.interestRateField;
            }
            set {
                this.interestRateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool interestRateSpecified {
            get {
                return this.interestRateFieldSpecified;
            }
            set {
                this.interestRateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public Money originalBalance {
            get {
                return this.originalBalanceField;
            }
            set {
                this.originalBalanceField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime maturityDate {
            get {
                return this.maturityDateField;
            }
            set {
                this.maturityDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool maturityDateSpecified {
            get {
                return this.maturityDateFieldSpecified;
            }
            set {
                this.maturityDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public Money escrowBalance {
            get {
                return this.escrowBalanceField;
            }
            set {
                this.escrowBalanceField = value;
            }
        }
        
        /// <remarks/>
        public Money minimumPayment {
            get {
                return this.minimumPaymentField;
            }
            set {
                this.minimumPaymentField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime currentDueDate {
            get {
                return this.currentDueDateField;
            }
            set {
                this.currentDueDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool currentDueDateSpecified {
            get {
                return this.currentDueDateFieldSpecified;
            }
            set {
                this.currentDueDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public Money currentPayoffBalance {
            get {
                return this.currentPayoffBalanceField;
            }
            set {
                this.currentPayoffBalanceField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(AnonymousType=true, Namespace="http://cufxstandards.com/v3/Transaction.xsd")]
    public partial class TransactionListTransaction {
        
        private string transactionIdField;
        
        private string accountIdField;
        
        private TransactionType typeField;
        
        private bool typeFieldSpecified;
        
        private Money amountField;
        
        private string descriptionField;
        
        private string checkNumberField;
        
        private System.DateTime dateTimePostedField;
        
        private bool dateTimePostedFieldSpecified;
        
        private System.DateTime dateTimeEffectiveField;
        
        private bool dateTimeEffectiveFieldSpecified;
        
        private TransactionStatus statusField;
        
        private Money principalAmountField;
        
        private Money interestAmountField;
        
        private TransactionFee[] transactionFeeListField;
        
        private string merchantCategoryCodeField;
        
        private string categoryField;
        
        private TransactionSource sourceField;
        
        private bool sourceFieldSpecified;
        
        private ValuePair[] customDataField;
        
        public TransactionListTransaction() {
            this.statusField = TransactionStatus.Posted;
        }
        
        /// <remarks/>
        public string transactionId {
            get {
                return this.transactionIdField;
            }
            set {
                this.transactionIdField = value;
            }
        }
        
        /// <remarks/>
        public string accountId {
            get {
                return this.accountIdField;
            }
            set {
                this.accountIdField = value;
            }
        }
        
        /// <remarks/>
        public TransactionType type {
            get {
                return this.typeField;
            }
            set {
                this.typeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool typeSpecified {
            get {
                return this.typeFieldSpecified;
            }
            set {
                this.typeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public Money amount {
            get {
                return this.amountField;
            }
            set {
                this.amountField = value;
            }
        }
        
        /// <remarks/>
        public string description {
            get {
                return this.descriptionField;
            }
            set {
                this.descriptionField = value;
            }
        }
        
        /// <remarks/>
        public string checkNumber {
            get {
                return this.checkNumberField;
            }
            set {
                this.checkNumberField = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime dateTimePosted {
            get {
                return this.dateTimePostedField;
            }
            set {
                this.dateTimePostedField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool dateTimePostedSpecified {
            get {
                return this.dateTimePostedFieldSpecified;
            }
            set {
                this.dateTimePostedFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime dateTimeEffective {
            get {
                return this.dateTimeEffectiveField;
            }
            set {
                this.dateTimeEffectiveField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool dateTimeEffectiveSpecified {
            get {
                return this.dateTimeEffectiveFieldSpecified;
            }
            set {
                this.dateTimeEffectiveFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.ComponentModel.DefaultValueAttribute(TransactionStatus.Posted)]
        public TransactionStatus status {
            get {
                return this.statusField;
            }
            set {
                this.statusField = value;
            }
        }
        
        /// <remarks/>
        public Money principalAmount {
            get {
                return this.principalAmountField;
            }
            set {
                this.principalAmountField = value;
            }
        }
        
        /// <remarks/>
        public Money interestAmount {
            get {
                return this.interestAmountField;
            }
            set {
                this.interestAmountField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("transactionFee", IsNullable=false)]
        public TransactionFee[] transactionFeeList {
            get {
                return this.transactionFeeListField;
            }
            set {
                this.transactionFeeListField = value;
            }
        }
        
        /// <remarks/>
        public string merchantCategoryCode {
            get {
                return this.merchantCategoryCodeField;
            }
            set {
                this.merchantCategoryCodeField = value;
            }
        }
        
        /// <remarks/>
        public string category {
            get {
                return this.categoryField;
            }
            set {
                this.categoryField = value;
            }
        }
        
        /// <remarks/>
        public TransactionSource source {
            get {
                return this.sourceField;
            }
            set {
                this.sourceField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool sourceSpecified {
            get {
                return this.sourceFieldSpecified;
            }
            set {
                this.sourceFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("valuePair", Namespace="http://cufxstandards.com/v3/Common.xsd", IsNullable=false)]
        public ValuePair[] customData {
            get {
                return this.customDataField;
            }
            set {
                this.customDataField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Transaction.xsd")]
    public enum TransactionType {
        
        /// <remarks/>
        Debit,
        
        /// <remarks/>
        Credit,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Transaction.xsd")]
    public enum TransactionStatus {
        
        /// <remarks/>
        Posted,
        
        /// <remarks/>
        Pending,
        
        /// <remarks/>
        Denied,
        
        /// <remarks/>
        Void,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Transaction.xsd")]
    public partial class TransactionFee {
        
        private string transactionFeeIdField;
        
        private System.DateTime transactionFeeDateTimePostedField;
        
        private bool transactionFeeDateTimePostedFieldSpecified;
        
        private Money transactionfeeAmountField;
        
        private string transactionFeeCodeField;
        
        private string transactionFeeDescriptionField;
        
        /// <remarks/>
        public string transactionFeeId {
            get {
                return this.transactionFeeIdField;
            }
            set {
                this.transactionFeeIdField = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime transactionFeeDateTimePosted {
            get {
                return this.transactionFeeDateTimePostedField;
            }
            set {
                this.transactionFeeDateTimePostedField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool transactionFeeDateTimePostedSpecified {
            get {
                return this.transactionFeeDateTimePostedFieldSpecified;
            }
            set {
                this.transactionFeeDateTimePostedFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public Money transactionfeeAmount {
            get {
                return this.transactionfeeAmountField;
            }
            set {
                this.transactionfeeAmountField = value;
            }
        }
        
        /// <remarks/>
        public string transactionFeeCode {
            get {
                return this.transactionFeeCodeField;
            }
            set {
                this.transactionFeeCodeField = value;
            }
        }
        
        /// <remarks/>
        public string transactionFeeDescription {
            get {
                return this.transactionFeeDescriptionField;
            }
            set {
                this.transactionFeeDescriptionField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Transaction.xsd")]
    public enum TransactionSource {
        
        /// <remarks/>
        Ach,
        
        /// <remarks/>
        Atm,
        
        /// <remarks/>
        BillPay,
        
        /// <remarks/>
        BulkDeposit,
        
        /// <remarks/>
        Cash,
        
        /// <remarks/>
        Check,
        
        /// <remarks/>
        Fee,
        
        /// <remarks/>
        HomeBanking,
        
        /// <remarks/>
        Insurance,
        
        /// <remarks/>
        InterestEarned,
        
        /// <remarks/>
        InterestPaid,
        
        /// <remarks/>
        Ivr,
        
        /// <remarks/>
        MobileBanking,
        
        /// <remarks/>
        Other,
        
        /// <remarks/>
        Payroll,
        
        /// <remarks/>
        PinPurchase,
        
        /// <remarks/>
        SharedBranch,
        
        /// <remarks/>
        Signature,
        
        /// <remarks/>
        Wire,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Loan.xsd")]
    public partial class Loan : Account {
        
        private LoanParty[] loanPartyListField;
        
        private string officerIdField;
        
        private string processorIdField;
        
        private System.DateTime loanDecisionDateField;
        
        private bool loanDecisionDateFieldSpecified;
        
        private System.DateTime applicationOriginationDateField;
        
        private bool applicationOriginationDateFieldSpecified;
        
        private LoanDecisionType applicationDecisionStatusField;
        
        private bool applicationDecisionStatusFieldSpecified;
        
        private LoanAccountStatus loanAccountStatusField;
        
        private bool loanAccountStatusFieldSpecified;
        
        private string loanAccountSubStatusField;
        
        private string noteNumberField;
        
        private LoanAccountCategory categoryField;
        
        private bool categoryFieldSpecified;
        
        private string purposeCodeField;
        
        private Money requestedAmountField;
        
        private Money noteAmountField;
        
        private Money creditLimitField;
        
        private System.DateTime creditLimitExpirationField;
        
        private bool creditLimitExpirationFieldSpecified;
        
        private string creditLimitGroupCodeField;
        
        private EcoaGroupType ecoaGroupField;
        
        private bool ecoaGroupFieldSpecified;
        
        private string creditReportingCodeField;
        
        private bool isRevolvingLineOfCreditField;
        
        private bool isRevolvingLineOfCreditFieldSpecified;
        
        private Money minimumAdvanceAmountField;
        
        private Money maximumAdvanceAmountField;
        
        private System.DateTime originationDateField;
        
        private bool originationDateFieldSpecified;
        
        private System.DateTime fundedDateField;
        
        private bool fundedDateFieldSpecified;
        
        private System.DateTime disbursalDateField;
        
        private bool disbursalDateFieldSpecified;
        
        private System.DateTime creationDateField;
        
        private bool creationDateFieldSpecified;
        
        private System.DateTime lastPaymentDateField;
        
        private bool lastPaymentDateFieldSpecified;
        
        private Money lastPaymentAmountField;
        
        private string termField;
        
        private LoanTermType termTypeField;
        
        private bool termTypeFieldSpecified;
        
        private CreditReport[] creditReportListField;
        
        private LoanInterestRateDetail loanInterestRateDetailField;
        
        private PaymentOption paymentOptionField;
        
        private LoanInterestCalculationType interestCalculationTypeField;
        
        private bool interestCalculationTypeFieldSpecified;
        
        private decimal dailyPeriodicRateField;
        
        private bool dailyPeriodicRateFieldSpecified;
        
        private string totalNumberOfPaymentsField;
        
        private Money prePaidFinanceChargesField;
        
        private Money prepaidInterestField;
        
        private System.DateTime interestPaidThruDateField;
        
        private bool interestPaidThruDateFieldSpecified;
        
        private string numberOfGraceDaysField;
        
        private Money balloonAmountField;
        
        private System.DateTime balloonDueDateField;
        
        private bool balloonDueDateFieldSpecified;
        
        private BalloonDueDateTermType balloonDueDateTermField;
        
        private bool balloonDueDateTermFieldSpecified;
        
        private string balloonDueDateTermValueField;
        
        private PrepaymentPenaltyBasisType prepaymentPenaltyBasisField;
        
        private bool prepaymentPenaltyBasisFieldSpecified;
        
        private Money prepaymentPenaltyBasisAmountField;
        
        private decimal prepaymentPenaltyPercentageField;
        
        private bool prepaymentPenaltyPercentageFieldSpecified;
        
        private System.DateTime prepaymentPenaltyExpirationDateField;
        
        private bool prepaymentPenaltyExpirationDateFieldSpecified;
        
        private string statementCodeField;
        
        private string statementGroupField;
        
        private System.DateTime statementDateField;
        
        private bool statementDateFieldSpecified;
        
        private decimal debtIncomeRatioField;
        
        private bool debtIncomeRatioFieldSpecified;
        
        private LoanFee[] loanFeeListField;
        
        private InsuranceAddOn[] insuranceAddOnListField;
        
        private Collateral[] collateralListField;
        
        private AutoPaymentOption autoPaymentOptionField;
        
        private DelinquencyNotice[] delinquencyNoticeListField;
        
        private SkipPayment skipPaymentField;
        
        private string promotionCodeField;
        
        private Note[] loanNoteListField;
        
        private CreditLimitIncreaseRequestList creditLimitIncreaseRequestListField;
        
        private ValuePair[] customData1Field;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("loanParty", IsNullable=false)]
        public LoanParty[] loanPartyList {
            get {
                return this.loanPartyListField;
            }
            set {
                this.loanPartyListField = value;
            }
        }
        
        /// <remarks/>
        public string officerId {
            get {
                return this.officerIdField;
            }
            set {
                this.officerIdField = value;
            }
        }
        
        /// <remarks/>
        public string processorId {
            get {
                return this.processorIdField;
            }
            set {
                this.processorIdField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime loanDecisionDate {
            get {
                return this.loanDecisionDateField;
            }
            set {
                this.loanDecisionDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool loanDecisionDateSpecified {
            get {
                return this.loanDecisionDateFieldSpecified;
            }
            set {
                this.loanDecisionDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime applicationOriginationDate {
            get {
                return this.applicationOriginationDateField;
            }
            set {
                this.applicationOriginationDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool applicationOriginationDateSpecified {
            get {
                return this.applicationOriginationDateFieldSpecified;
            }
            set {
                this.applicationOriginationDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public LoanDecisionType applicationDecisionStatus {
            get {
                return this.applicationDecisionStatusField;
            }
            set {
                this.applicationDecisionStatusField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool applicationDecisionStatusSpecified {
            get {
                return this.applicationDecisionStatusFieldSpecified;
            }
            set {
                this.applicationDecisionStatusFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public LoanAccountStatus loanAccountStatus {
            get {
                return this.loanAccountStatusField;
            }
            set {
                this.loanAccountStatusField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool loanAccountStatusSpecified {
            get {
                return this.loanAccountStatusFieldSpecified;
            }
            set {
                this.loanAccountStatusFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string loanAccountSubStatus {
            get {
                return this.loanAccountSubStatusField;
            }
            set {
                this.loanAccountSubStatusField = value;
            }
        }
        
        /// <remarks/>
        public string noteNumber {
            get {
                return this.noteNumberField;
            }
            set {
                this.noteNumberField = value;
            }
        }
        
        /// <remarks/>
        public LoanAccountCategory category {
            get {
                return this.categoryField;
            }
            set {
                this.categoryField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool categorySpecified {
            get {
                return this.categoryFieldSpecified;
            }
            set {
                this.categoryFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string purposeCode {
            get {
                return this.purposeCodeField;
            }
            set {
                this.purposeCodeField = value;
            }
        }
        
        /// <remarks/>
        public Money requestedAmount {
            get {
                return this.requestedAmountField;
            }
            set {
                this.requestedAmountField = value;
            }
        }
        
        /// <remarks/>
        public Money noteAmount {
            get {
                return this.noteAmountField;
            }
            set {
                this.noteAmountField = value;
            }
        }
        
        /// <remarks/>
        public Money creditLimit {
            get {
                return this.creditLimitField;
            }
            set {
                this.creditLimitField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime creditLimitExpiration {
            get {
                return this.creditLimitExpirationField;
            }
            set {
                this.creditLimitExpirationField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool creditLimitExpirationSpecified {
            get {
                return this.creditLimitExpirationFieldSpecified;
            }
            set {
                this.creditLimitExpirationFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string creditLimitGroupCode {
            get {
                return this.creditLimitGroupCodeField;
            }
            set {
                this.creditLimitGroupCodeField = value;
            }
        }
        
        /// <remarks/>
        public EcoaGroupType ecoaGroup {
            get {
                return this.ecoaGroupField;
            }
            set {
                this.ecoaGroupField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool ecoaGroupSpecified {
            get {
                return this.ecoaGroupFieldSpecified;
            }
            set {
                this.ecoaGroupFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string creditReportingCode {
            get {
                return this.creditReportingCodeField;
            }
            set {
                this.creditReportingCodeField = value;
            }
        }
        
        /// <remarks/>
        public bool isRevolvingLineOfCredit {
            get {
                return this.isRevolvingLineOfCreditField;
            }
            set {
                this.isRevolvingLineOfCreditField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool isRevolvingLineOfCreditSpecified {
            get {
                return this.isRevolvingLineOfCreditFieldSpecified;
            }
            set {
                this.isRevolvingLineOfCreditFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public Money minimumAdvanceAmount {
            get {
                return this.minimumAdvanceAmountField;
            }
            set {
                this.minimumAdvanceAmountField = value;
            }
        }
        
        /// <remarks/>
        public Money maximumAdvanceAmount {
            get {
                return this.maximumAdvanceAmountField;
            }
            set {
                this.maximumAdvanceAmountField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime originationDate {
            get {
                return this.originationDateField;
            }
            set {
                this.originationDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool originationDateSpecified {
            get {
                return this.originationDateFieldSpecified;
            }
            set {
                this.originationDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime fundedDate {
            get {
                return this.fundedDateField;
            }
            set {
                this.fundedDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool fundedDateSpecified {
            get {
                return this.fundedDateFieldSpecified;
            }
            set {
                this.fundedDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime disbursalDate {
            get {
                return this.disbursalDateField;
            }
            set {
                this.disbursalDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool disbursalDateSpecified {
            get {
                return this.disbursalDateFieldSpecified;
            }
            set {
                this.disbursalDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime creationDate {
            get {
                return this.creationDateField;
            }
            set {
                this.creationDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool creationDateSpecified {
            get {
                return this.creationDateFieldSpecified;
            }
            set {
                this.creationDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime lastPaymentDate {
            get {
                return this.lastPaymentDateField;
            }
            set {
                this.lastPaymentDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool lastPaymentDateSpecified {
            get {
                return this.lastPaymentDateFieldSpecified;
            }
            set {
                this.lastPaymentDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public Money lastPaymentAmount {
            get {
                return this.lastPaymentAmountField;
            }
            set {
                this.lastPaymentAmountField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="integer")]
        public string term {
            get {
                return this.termField;
            }
            set {
                this.termField = value;
            }
        }
        
        /// <remarks/>
        public LoanTermType termType {
            get {
                return this.termTypeField;
            }
            set {
                this.termTypeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool termTypeSpecified {
            get {
                return this.termTypeFieldSpecified;
            }
            set {
                this.termTypeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("creditReport", Namespace="http://cufxstandards.com/v3/CreditReport.xsd", IsNullable=false)]
        public CreditReport[] creditReportList {
            get {
                return this.creditReportListField;
            }
            set {
                this.creditReportListField = value;
            }
        }
        
        /// <remarks/>
        public LoanInterestRateDetail loanInterestRateDetail {
            get {
                return this.loanInterestRateDetailField;
            }
            set {
                this.loanInterestRateDetailField = value;
            }
        }
        
        /// <remarks/>
        public PaymentOption paymentOption {
            get {
                return this.paymentOptionField;
            }
            set {
                this.paymentOptionField = value;
            }
        }
        
        /// <remarks/>
        public LoanInterestCalculationType interestCalculationType {
            get {
                return this.interestCalculationTypeField;
            }
            set {
                this.interestCalculationTypeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool interestCalculationTypeSpecified {
            get {
                return this.interestCalculationTypeFieldSpecified;
            }
            set {
                this.interestCalculationTypeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public decimal dailyPeriodicRate {
            get {
                return this.dailyPeriodicRateField;
            }
            set {
                this.dailyPeriodicRateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool dailyPeriodicRateSpecified {
            get {
                return this.dailyPeriodicRateFieldSpecified;
            }
            set {
                this.dailyPeriodicRateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="integer")]
        public string totalNumberOfPayments {
            get {
                return this.totalNumberOfPaymentsField;
            }
            set {
                this.totalNumberOfPaymentsField = value;
            }
        }
        
        /// <remarks/>
        public Money prePaidFinanceCharges {
            get {
                return this.prePaidFinanceChargesField;
            }
            set {
                this.prePaidFinanceChargesField = value;
            }
        }
        
        /// <remarks/>
        public Money prepaidInterest {
            get {
                return this.prepaidInterestField;
            }
            set {
                this.prepaidInterestField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime interestPaidThruDate {
            get {
                return this.interestPaidThruDateField;
            }
            set {
                this.interestPaidThruDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool interestPaidThruDateSpecified {
            get {
                return this.interestPaidThruDateFieldSpecified;
            }
            set {
                this.interestPaidThruDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="integer")]
        public string numberOfGraceDays {
            get {
                return this.numberOfGraceDaysField;
            }
            set {
                this.numberOfGraceDaysField = value;
            }
        }
        
        /// <remarks/>
        public Money balloonAmount {
            get {
                return this.balloonAmountField;
            }
            set {
                this.balloonAmountField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime balloonDueDate {
            get {
                return this.balloonDueDateField;
            }
            set {
                this.balloonDueDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool balloonDueDateSpecified {
            get {
                return this.balloonDueDateFieldSpecified;
            }
            set {
                this.balloonDueDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public BalloonDueDateTermType balloonDueDateTerm {
            get {
                return this.balloonDueDateTermField;
            }
            set {
                this.balloonDueDateTermField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool balloonDueDateTermSpecified {
            get {
                return this.balloonDueDateTermFieldSpecified;
            }
            set {
                this.balloonDueDateTermFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="integer")]
        public string balloonDueDateTermValue {
            get {
                return this.balloonDueDateTermValueField;
            }
            set {
                this.balloonDueDateTermValueField = value;
            }
        }
        
        /// <remarks/>
        public PrepaymentPenaltyBasisType prepaymentPenaltyBasis {
            get {
                return this.prepaymentPenaltyBasisField;
            }
            set {
                this.prepaymentPenaltyBasisField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool prepaymentPenaltyBasisSpecified {
            get {
                return this.prepaymentPenaltyBasisFieldSpecified;
            }
            set {
                this.prepaymentPenaltyBasisFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public Money prepaymentPenaltyBasisAmount {
            get {
                return this.prepaymentPenaltyBasisAmountField;
            }
            set {
                this.prepaymentPenaltyBasisAmountField = value;
            }
        }
        
        /// <remarks/>
        public decimal prepaymentPenaltyPercentage {
            get {
                return this.prepaymentPenaltyPercentageField;
            }
            set {
                this.prepaymentPenaltyPercentageField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool prepaymentPenaltyPercentageSpecified {
            get {
                return this.prepaymentPenaltyPercentageFieldSpecified;
            }
            set {
                this.prepaymentPenaltyPercentageFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime prepaymentPenaltyExpirationDate {
            get {
                return this.prepaymentPenaltyExpirationDateField;
            }
            set {
                this.prepaymentPenaltyExpirationDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool prepaymentPenaltyExpirationDateSpecified {
            get {
                return this.prepaymentPenaltyExpirationDateFieldSpecified;
            }
            set {
                this.prepaymentPenaltyExpirationDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string statementCode {
            get {
                return this.statementCodeField;
            }
            set {
                this.statementCodeField = value;
            }
        }
        
        /// <remarks/>
        public string statementGroup {
            get {
                return this.statementGroupField;
            }
            set {
                this.statementGroupField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime statementDate {
            get {
                return this.statementDateField;
            }
            set {
                this.statementDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool statementDateSpecified {
            get {
                return this.statementDateFieldSpecified;
            }
            set {
                this.statementDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public decimal debtIncomeRatio {
            get {
                return this.debtIncomeRatioField;
            }
            set {
                this.debtIncomeRatioField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool debtIncomeRatioSpecified {
            get {
                return this.debtIncomeRatioFieldSpecified;
            }
            set {
                this.debtIncomeRatioFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("loanFee", IsNullable=false)]
        public LoanFee[] loanFeeList {
            get {
                return this.loanFeeListField;
            }
            set {
                this.loanFeeListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("insuranceAddOn", IsNullable=false)]
        public InsuranceAddOn[] insuranceAddOnList {
            get {
                return this.insuranceAddOnListField;
            }
            set {
                this.insuranceAddOnListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("collateral", Namespace="http://cufxstandards.com/v3/Collateral.xsd", IsNullable=false)]
        public Collateral[] collateralList {
            get {
                return this.collateralListField;
            }
            set {
                this.collateralListField = value;
            }
        }
        
        /// <remarks/>
        public AutoPaymentOption autoPaymentOption {
            get {
                return this.autoPaymentOptionField;
            }
            set {
                this.autoPaymentOptionField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("delinquencyNotice", IsNullable=false)]
        public DelinquencyNotice[] delinquencyNoticeList {
            get {
                return this.delinquencyNoticeListField;
            }
            set {
                this.delinquencyNoticeListField = value;
            }
        }
        
        /// <remarks/>
        public SkipPayment skipPayment {
            get {
                return this.skipPaymentField;
            }
            set {
                this.skipPaymentField = value;
            }
        }
        
        /// <remarks/>
        public string promotionCode {
            get {
                return this.promotionCodeField;
            }
            set {
                this.promotionCodeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("note", Namespace="http://cufxstandards.com/v3/Common.xsd", IsNullable=false)]
        public Note[] loanNoteList {
            get {
                return this.loanNoteListField;
            }
            set {
                this.loanNoteListField = value;
            }
        }
        
        /// <remarks/>
        public CreditLimitIncreaseRequestList creditLimitIncreaseRequestList {
            get {
                return this.creditLimitIncreaseRequestListField;
            }
            set {
                this.creditLimitIncreaseRequestListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayAttribute("customData")]
        [System.Xml.Serialization.XmlArrayItemAttribute("valuePair", Namespace="http://cufxstandards.com/v3/Common.xsd", IsNullable=false)]
        public ValuePair[] customData1 {
            get {
                return this.customData1Field;
            }
            set {
                this.customData1Field = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Loan.xsd")]
    public partial class LoanParty {
        
        private string loanPartyIdField;
        
        private LoanPartyRelationshipType loanPartyRelationshipTypeField;
        
        private string[] contactIdListField;
        
        /// <remarks/>
        public string loanPartyId {
            get {
                return this.loanPartyIdField;
            }
            set {
                this.loanPartyIdField = value;
            }
        }
        
        /// <remarks/>
        public LoanPartyRelationshipType loanPartyRelationshipType {
            get {
                return this.loanPartyRelationshipTypeField;
            }
            set {
                this.loanPartyRelationshipTypeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("contactId", Namespace="http://cufxstandards.com/v3/Contact.xsd", IsNullable=false)]
        public string[] contactIdList {
            get {
                return this.contactIdListField;
            }
            set {
                this.contactIdListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Loan.xsd")]
    public partial class LoanPartyRelationshipType {
        
        private object itemField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("agent", typeof(Agent))]
        [System.Xml.Serialization.XmlElementAttribute("borrower", typeof(Borrower))]
        [System.Xml.Serialization.XmlElementAttribute("collateralGrantor", typeof(CollateralGrantor))]
        [System.Xml.Serialization.XmlElementAttribute("guarantor", typeof(Guarantor))]
        [System.Xml.Serialization.XmlElementAttribute("payee", typeof(Payee))]
        public object Item {
            get {
                return this.itemField;
            }
            set {
                this.itemField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Common.xsd")]
    public partial class Agent {
        
        private AgentQualifier qualifierField;
        
        private Authority authorityField;
        
        public Agent() {
            this.authorityField = Authority.Unauthorized;
        }
        
        /// <remarks/>
        public AgentQualifier qualifier {
            get {
                return this.qualifierField;
            }
            set {
                this.qualifierField = value;
            }
        }
        
        /// <remarks/>
        public Authority authority {
            get {
                return this.authorityField;
            }
            set {
                this.authorityField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Common.xsd")]
    public enum AgentQualifier {
        
        /// <remarks/>
        Custodian,
        
        /// <remarks/>
        Trustee,
        
        /// <remarks/>
        PowerOfAttorney,
        
        /// <remarks/>
        Representative,
        
        /// <remarks/>
        CtrTransactor,
        
        /// <remarks/>
        AttorneyTrust,
        
        /// <remarks/>
        ResponsibleIndivfidual,
        
        /// <remarks/>
        Signer,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Common.xsd")]
    public enum Authority {
        
        /// <remarks/>
        Authorized,
        
        /// <remarks/>
        Unauthorized,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Loan.xsd")]
    public partial class Borrower {
        
        private PrimaryJoint qualifierField;
        
        private Authority authorityField;
        
        /// <remarks/>
        public PrimaryJoint qualifier {
            get {
                return this.qualifierField;
            }
            set {
                this.qualifierField = value;
            }
        }
        
        /// <remarks/>
        public Authority authority {
            get {
                return this.authorityField;
            }
            set {
                this.authorityField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Relationship.xsd")]
    public enum PrimaryJoint {
        
        /// <remarks/>
        Primary,
        
        /// <remarks/>
        Joint,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Relationship.xsd")]
    public partial class CollateralGrantor {
        
        private Authority authorityField;
        
        public CollateralGrantor() {
            this.authorityField = Authority.Unauthorized;
        }
        
        /// <remarks/>
        public Authority authority {
            get {
                return this.authorityField;
            }
            set {
                this.authorityField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Relationship.xsd")]
    public partial class Guarantor {
        
        private Authority authorityField;
        
        /// <remarks/>
        public Authority authority {
            get {
                return this.authorityField;
            }
            set {
                this.authorityField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Relationship.xsd")]
    public partial class Payee {
        
        private PayeeQualifier qualifierField;
        
        private string authorityField;
        
        public Payee() {
            this.authorityField = "DIVIDEND_ONLY";
        }
        
        /// <remarks/>
        public PayeeQualifier qualifier {
            get {
                return this.qualifierField;
            }
            set {
                this.qualifierField = value;
            }
        }
        
        /// <remarks/>
        public string authority {
            get {
                return this.authorityField;
            }
            set {
                this.authorityField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Relationship.xsd")]
    public enum PayeeQualifier {
        
        /// <remarks/>
        Dividend,
        
        /// <remarks/>
        Maturity,
        
        /// <remarks/>
        DividendAndMaturity,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Loan.xsd")]
    public enum LoanDecisionType {
        
        /// <remarks/>
        Approve,
        
        /// <remarks/>
        CounterOffer,
        
        /// <remarks/>
        Denied,
        
        /// <remarks/>
        Review,
        
        /// <remarks/>
        Withdrawn,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Loan.xsd")]
    public enum LoanAccountStatus {
        
        /// <remarks/>
        Active,
        
        /// <remarks/>
        Closed,
        
        /// <remarks/>
        Dormant,
        
        /// <remarks/>
        Escheated,
        
        /// <remarks/>
        Incomplete,
        
        /// <remarks/>
        Locked,
        
        /// <remarks/>
        Matured,
        
        /// <remarks/>
        RenewPending,
        
        /// <remarks/>
        Restricted,
        
        /// <remarks/>
        Unfunded,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Loan.xsd")]
    public enum LoanAccountCategory {
        
        /// <remarks/>
        AverageDailyBalanceLineOfCredit,
        
        /// <remarks/>
        ClosedEnd,
        
        /// <remarks/>
        CreditCard,
        
        /// <remarks/>
        Lease,
        
        /// <remarks/>
        LineOfCredit,
        
        /// <remarks/>
        LineOfCreditCombination,
        
        /// <remarks/>
        OpenEnd,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Loan.xsd")]
    public enum EcoaGroupType {
        
        /// <remarks/>
        Individual,
        
        /// <remarks/>
        Joint,
        
        /// <remarks/>
        AuthorizedUser,
        
        /// <remarks/>
        CoSigner,
        
        /// <remarks/>
        Maker,
        
        /// <remarks/>
        CoMaker,
        
        /// <remarks/>
        Terminated,
        
        /// <remarks/>
        Undesignated,
        
        /// <remarks/>
        BusinessOrCommercial,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Loan.xsd")]
    public enum LoanTermType {
        
        /// <remarks/>
        Days,
        
        /// <remarks/>
        Months,
        
        /// <remarks/>
        Years,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/CreditReport.xsd")]
    public partial class CreditReport {
        
        private string creditReportIdField;
        
        private System.DateTime creditReportDateField;
        
        private bool creditReportDateFieldSpecified;
        
        private string creditSourceField;
        
        private string reportTypeField;
        
        private string scoreTypeField;
        
        private string partyIdField;
        
        private string taxIdField;
        
        private string creditScoreField;
        
        private string creditTierField;
        
        private string reportDataField;
        
        /// <remarks/>
        public string creditReportId {
            get {
                return this.creditReportIdField;
            }
            set {
                this.creditReportIdField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime creditReportDate {
            get {
                return this.creditReportDateField;
            }
            set {
                this.creditReportDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool creditReportDateSpecified {
            get {
                return this.creditReportDateFieldSpecified;
            }
            set {
                this.creditReportDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string creditSource {
            get {
                return this.creditSourceField;
            }
            set {
                this.creditSourceField = value;
            }
        }
        
        /// <remarks/>
        public string reportType {
            get {
                return this.reportTypeField;
            }
            set {
                this.reportTypeField = value;
            }
        }
        
        /// <remarks/>
        public string scoreType {
            get {
                return this.scoreTypeField;
            }
            set {
                this.scoreTypeField = value;
            }
        }
        
        /// <remarks/>
        public string partyId {
            get {
                return this.partyIdField;
            }
            set {
                this.partyIdField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="token")]
        public string taxId {
            get {
                return this.taxIdField;
            }
            set {
                this.taxIdField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="integer")]
        public string creditScore {
            get {
                return this.creditScoreField;
            }
            set {
                this.creditScoreField = value;
            }
        }
        
        /// <remarks/>
        public string creditTier {
            get {
                return this.creditTierField;
            }
            set {
                this.creditTierField = value;
            }
        }
        
        /// <remarks/>
        public string reportData {
            get {
                return this.reportDataField;
            }
            set {
                this.reportDataField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Loan.xsd")]
    public partial class LoanInterestRateDetail {
        
        private decimal originalEffectiveRateField;
        
        private bool originalEffectiveRateFieldSpecified;
        
        private decimal rateDiscountPercentageField;
        
        private bool rateDiscountPercentageFieldSpecified;
        
        private string rateDiscountReasonCodeField;
        
        private decimal relationshipRateDiscountPercentageField;
        
        private bool relationshipRateDiscountPercentageFieldSpecified;
        
        private string relationshipRateDiscountReasonCodeField;
        
        private decimal annualPercentageRateField;
        
        private bool annualPercentageRateFieldSpecified;
        
        private decimal rateField;
        
        private bool rateFieldSpecified;
        
        private bool isSplitRateField;
        
        private bool isSplitRateFieldSpecified;
        
        private string splitRateCodeField;
        
        private decimal rateMaximumField;
        
        private bool rateMaximumFieldSpecified;
        
        private decimal rateMinimumField;
        
        private bool rateMinimumFieldSpecified;
        
        private decimal ratePercentageChangeField;
        
        private bool ratePercentageChangeFieldSpecified;
        
        private InterestRateMarginType marginField;
        
        private bool marginFieldSpecified;
        
        private decimal baseRateIndexField;
        
        private bool baseRateIndexFieldSpecified;
        
        private decimal rateMarginPercentageField;
        
        private bool rateMarginPercentageFieldSpecified;
        
        private decimal maximumFirstRateChangeIncreaseField;
        
        private bool maximumFirstRateChangeIncreaseFieldSpecified;
        
        private decimal maximumFirstRateChangeDecreaseField;
        
        private bool maximumFirstRateChangeDecreaseFieldSpecified;
        
        private decimal maximumPercentageAdjustableRateChangeIncreaseField;
        
        private bool maximumPercentageAdjustableRateChangeIncreaseFieldSpecified;
        
        private decimal maxPercentageAdjustableRateChangeDecreaseField;
        
        private bool maxPercentageAdjustableRateChangeDecreaseFieldSpecified;
        
        private decimal maxAnnualRateChangeIncreaseField;
        
        private bool maxAnnualRateChangeIncreaseFieldSpecified;
        
        private decimal maxAnnualRateChangeDecreaseField;
        
        private bool maxAnnualRateChangeDecreaseFieldSpecified;
        
        private bool isRateMarginAboveIndexField;
        
        private bool isRateMarginAboveIndexFieldSpecified;
        
        private decimal marginRiskRateField;
        
        private bool marginRiskRateFieldSpecified;
        
        private System.DateTime scheduledRateChangeDateField;
        
        private bool scheduledRateChangeDateFieldSpecified;
        
        private Money periodicCapAmountField;
        
        private decimal periodicCapStartRateField;
        
        private bool periodicCapStartRateFieldSpecified;
        
        private System.DateTime periodicCapStartDateField;
        
        private bool periodicCapStartDateFieldSpecified;
        
        private PeriodicCapFrequencyType periodicCapFrequencyField;
        
        private bool periodicCapFrequencyFieldSpecified;
        
        private string periodicCapFrequencyValueField;
        
        private Money splitRateBalanceField;
        
        /// <remarks/>
        public decimal originalEffectiveRate {
            get {
                return this.originalEffectiveRateField;
            }
            set {
                this.originalEffectiveRateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool originalEffectiveRateSpecified {
            get {
                return this.originalEffectiveRateFieldSpecified;
            }
            set {
                this.originalEffectiveRateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public decimal rateDiscountPercentage {
            get {
                return this.rateDiscountPercentageField;
            }
            set {
                this.rateDiscountPercentageField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool rateDiscountPercentageSpecified {
            get {
                return this.rateDiscountPercentageFieldSpecified;
            }
            set {
                this.rateDiscountPercentageFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string rateDiscountReasonCode {
            get {
                return this.rateDiscountReasonCodeField;
            }
            set {
                this.rateDiscountReasonCodeField = value;
            }
        }
        
        /// <remarks/>
        public decimal relationshipRateDiscountPercentage {
            get {
                return this.relationshipRateDiscountPercentageField;
            }
            set {
                this.relationshipRateDiscountPercentageField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool relationshipRateDiscountPercentageSpecified {
            get {
                return this.relationshipRateDiscountPercentageFieldSpecified;
            }
            set {
                this.relationshipRateDiscountPercentageFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string relationshipRateDiscountReasonCode {
            get {
                return this.relationshipRateDiscountReasonCodeField;
            }
            set {
                this.relationshipRateDiscountReasonCodeField = value;
            }
        }
        
        /// <remarks/>
        public decimal annualPercentageRate {
            get {
                return this.annualPercentageRateField;
            }
            set {
                this.annualPercentageRateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool annualPercentageRateSpecified {
            get {
                return this.annualPercentageRateFieldSpecified;
            }
            set {
                this.annualPercentageRateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public decimal rate {
            get {
                return this.rateField;
            }
            set {
                this.rateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool rateSpecified {
            get {
                return this.rateFieldSpecified;
            }
            set {
                this.rateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public bool isSplitRate {
            get {
                return this.isSplitRateField;
            }
            set {
                this.isSplitRateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool isSplitRateSpecified {
            get {
                return this.isSplitRateFieldSpecified;
            }
            set {
                this.isSplitRateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string splitRateCode {
            get {
                return this.splitRateCodeField;
            }
            set {
                this.splitRateCodeField = value;
            }
        }
        
        /// <remarks/>
        public decimal rateMaximum {
            get {
                return this.rateMaximumField;
            }
            set {
                this.rateMaximumField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool rateMaximumSpecified {
            get {
                return this.rateMaximumFieldSpecified;
            }
            set {
                this.rateMaximumFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public decimal rateMinimum {
            get {
                return this.rateMinimumField;
            }
            set {
                this.rateMinimumField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool rateMinimumSpecified {
            get {
                return this.rateMinimumFieldSpecified;
            }
            set {
                this.rateMinimumFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public decimal ratePercentageChange {
            get {
                return this.ratePercentageChangeField;
            }
            set {
                this.ratePercentageChangeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool ratePercentageChangeSpecified {
            get {
                return this.ratePercentageChangeFieldSpecified;
            }
            set {
                this.ratePercentageChangeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public InterestRateMarginType margin {
            get {
                return this.marginField;
            }
            set {
                this.marginField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool marginSpecified {
            get {
                return this.marginFieldSpecified;
            }
            set {
                this.marginFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public decimal baseRateIndex {
            get {
                return this.baseRateIndexField;
            }
            set {
                this.baseRateIndexField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool baseRateIndexSpecified {
            get {
                return this.baseRateIndexFieldSpecified;
            }
            set {
                this.baseRateIndexFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public decimal rateMarginPercentage {
            get {
                return this.rateMarginPercentageField;
            }
            set {
                this.rateMarginPercentageField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool rateMarginPercentageSpecified {
            get {
                return this.rateMarginPercentageFieldSpecified;
            }
            set {
                this.rateMarginPercentageFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public decimal maximumFirstRateChangeIncrease {
            get {
                return this.maximumFirstRateChangeIncreaseField;
            }
            set {
                this.maximumFirstRateChangeIncreaseField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool maximumFirstRateChangeIncreaseSpecified {
            get {
                return this.maximumFirstRateChangeIncreaseFieldSpecified;
            }
            set {
                this.maximumFirstRateChangeIncreaseFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public decimal maximumFirstRateChangeDecrease {
            get {
                return this.maximumFirstRateChangeDecreaseField;
            }
            set {
                this.maximumFirstRateChangeDecreaseField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool maximumFirstRateChangeDecreaseSpecified {
            get {
                return this.maximumFirstRateChangeDecreaseFieldSpecified;
            }
            set {
                this.maximumFirstRateChangeDecreaseFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public decimal maximumPercentageAdjustableRateChangeIncrease {
            get {
                return this.maximumPercentageAdjustableRateChangeIncreaseField;
            }
            set {
                this.maximumPercentageAdjustableRateChangeIncreaseField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool maximumPercentageAdjustableRateChangeIncreaseSpecified {
            get {
                return this.maximumPercentageAdjustableRateChangeIncreaseFieldSpecified;
            }
            set {
                this.maximumPercentageAdjustableRateChangeIncreaseFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public decimal maxPercentageAdjustableRateChangeDecrease {
            get {
                return this.maxPercentageAdjustableRateChangeDecreaseField;
            }
            set {
                this.maxPercentageAdjustableRateChangeDecreaseField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool maxPercentageAdjustableRateChangeDecreaseSpecified {
            get {
                return this.maxPercentageAdjustableRateChangeDecreaseFieldSpecified;
            }
            set {
                this.maxPercentageAdjustableRateChangeDecreaseFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public decimal maxAnnualRateChangeIncrease {
            get {
                return this.maxAnnualRateChangeIncreaseField;
            }
            set {
                this.maxAnnualRateChangeIncreaseField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool maxAnnualRateChangeIncreaseSpecified {
            get {
                return this.maxAnnualRateChangeIncreaseFieldSpecified;
            }
            set {
                this.maxAnnualRateChangeIncreaseFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public decimal maxAnnualRateChangeDecrease {
            get {
                return this.maxAnnualRateChangeDecreaseField;
            }
            set {
                this.maxAnnualRateChangeDecreaseField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool maxAnnualRateChangeDecreaseSpecified {
            get {
                return this.maxAnnualRateChangeDecreaseFieldSpecified;
            }
            set {
                this.maxAnnualRateChangeDecreaseFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public bool isRateMarginAboveIndex {
            get {
                return this.isRateMarginAboveIndexField;
            }
            set {
                this.isRateMarginAboveIndexField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool isRateMarginAboveIndexSpecified {
            get {
                return this.isRateMarginAboveIndexFieldSpecified;
            }
            set {
                this.isRateMarginAboveIndexFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public decimal marginRiskRate {
            get {
                return this.marginRiskRateField;
            }
            set {
                this.marginRiskRateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool marginRiskRateSpecified {
            get {
                return this.marginRiskRateFieldSpecified;
            }
            set {
                this.marginRiskRateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime scheduledRateChangeDate {
            get {
                return this.scheduledRateChangeDateField;
            }
            set {
                this.scheduledRateChangeDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool scheduledRateChangeDateSpecified {
            get {
                return this.scheduledRateChangeDateFieldSpecified;
            }
            set {
                this.scheduledRateChangeDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public Money periodicCapAmount {
            get {
                return this.periodicCapAmountField;
            }
            set {
                this.periodicCapAmountField = value;
            }
        }
        
        /// <remarks/>
        public decimal periodicCapStartRate {
            get {
                return this.periodicCapStartRateField;
            }
            set {
                this.periodicCapStartRateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool periodicCapStartRateSpecified {
            get {
                return this.periodicCapStartRateFieldSpecified;
            }
            set {
                this.periodicCapStartRateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime periodicCapStartDate {
            get {
                return this.periodicCapStartDateField;
            }
            set {
                this.periodicCapStartDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool periodicCapStartDateSpecified {
            get {
                return this.periodicCapStartDateFieldSpecified;
            }
            set {
                this.periodicCapStartDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public PeriodicCapFrequencyType periodicCapFrequency {
            get {
                return this.periodicCapFrequencyField;
            }
            set {
                this.periodicCapFrequencyField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool periodicCapFrequencySpecified {
            get {
                return this.periodicCapFrequencyFieldSpecified;
            }
            set {
                this.periodicCapFrequencyFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="integer")]
        public string periodicCapFrequencyValue {
            get {
                return this.periodicCapFrequencyValueField;
            }
            set {
                this.periodicCapFrequencyValueField = value;
            }
        }
        
        /// <remarks/>
        public Money splitRateBalance {
            get {
                return this.splitRateBalanceField;
            }
            set {
                this.splitRateBalanceField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Loan.xsd")]
    public enum InterestRateMarginType {
        
        /// <remarks/>
        Percentage,
        
        /// <remarks/>
        Points,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Loan.xsd")]
    public enum PeriodicCapFrequencyType {
        
        /// <remarks/>
        Annual,
        
        /// <remarks/>
        Quarterly,
        
        /// <remarks/>
        Monthly,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("Semi-Annual")]
        SemiAnnual,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Loan.xsd")]
    public partial class PaymentOption {
        
        private string calculationCodeField;
        
        private PaymentOptionType typeField;
        
        private bool typeFieldSpecified;
        
        private string calculationMethodField;
        
        private decimal calculationPercentageField;
        
        private bool calculationPercentageFieldSpecified;
        
        private string paymentApplicationOrderCodeField;
        
        private LoanPaymentFrequencyType frequencyField;
        
        private bool frequencyFieldSpecified;
        
        private string numberOfAnnualPaymentsField;
        
        private string adjustmentMethodField;
        
        private bool isRoundPaymentField;
        
        private bool isRoundPaymentFieldSpecified;
        
        private Money paymentAmountField;
        
        private System.DateTime dueDateField;
        
        private bool dueDateFieldSpecified;
        
        private System.DateTime firstPaymentDueDateField;
        
        private bool firstPaymentDueDateFieldSpecified;
        
        private Money minimumPaymentAmountField;
        
        private Money latePaymentAmountField;
        
        private Money finalPaymentAmountField;
        
        private System.DateTime maturityDateField;
        
        private bool maturityDateFieldSpecified;
        
        private string paymentDayField;
        
        private LoanPaymentMethodType paymentMethodField;
        
        private bool paymentMethodFieldSpecified;
        
        private string otherPaymentMethodField;
        
        private string couponCodeField;
        
        private Money amountIncludingPrincipalAndInterestField;
        
        /// <remarks/>
        public string calculationCode {
            get {
                return this.calculationCodeField;
            }
            set {
                this.calculationCodeField = value;
            }
        }
        
        /// <remarks/>
        public PaymentOptionType type {
            get {
                return this.typeField;
            }
            set {
                this.typeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool typeSpecified {
            get {
                return this.typeFieldSpecified;
            }
            set {
                this.typeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string calculationMethod {
            get {
                return this.calculationMethodField;
            }
            set {
                this.calculationMethodField = value;
            }
        }
        
        /// <remarks/>
        public decimal calculationPercentage {
            get {
                return this.calculationPercentageField;
            }
            set {
                this.calculationPercentageField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool calculationPercentageSpecified {
            get {
                return this.calculationPercentageFieldSpecified;
            }
            set {
                this.calculationPercentageFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string paymentApplicationOrderCode {
            get {
                return this.paymentApplicationOrderCodeField;
            }
            set {
                this.paymentApplicationOrderCodeField = value;
            }
        }
        
        /// <remarks/>
        public LoanPaymentFrequencyType frequency {
            get {
                return this.frequencyField;
            }
            set {
                this.frequencyField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool frequencySpecified {
            get {
                return this.frequencyFieldSpecified;
            }
            set {
                this.frequencyFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="integer")]
        public string numberOfAnnualPayments {
            get {
                return this.numberOfAnnualPaymentsField;
            }
            set {
                this.numberOfAnnualPaymentsField = value;
            }
        }
        
        /// <remarks/>
        public string adjustmentMethod {
            get {
                return this.adjustmentMethodField;
            }
            set {
                this.adjustmentMethodField = value;
            }
        }
        
        /// <remarks/>
        public bool isRoundPayment {
            get {
                return this.isRoundPaymentField;
            }
            set {
                this.isRoundPaymentField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool isRoundPaymentSpecified {
            get {
                return this.isRoundPaymentFieldSpecified;
            }
            set {
                this.isRoundPaymentFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public Money paymentAmount {
            get {
                return this.paymentAmountField;
            }
            set {
                this.paymentAmountField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime dueDate {
            get {
                return this.dueDateField;
            }
            set {
                this.dueDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool dueDateSpecified {
            get {
                return this.dueDateFieldSpecified;
            }
            set {
                this.dueDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime firstPaymentDueDate {
            get {
                return this.firstPaymentDueDateField;
            }
            set {
                this.firstPaymentDueDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool firstPaymentDueDateSpecified {
            get {
                return this.firstPaymentDueDateFieldSpecified;
            }
            set {
                this.firstPaymentDueDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public Money minimumPaymentAmount {
            get {
                return this.minimumPaymentAmountField;
            }
            set {
                this.minimumPaymentAmountField = value;
            }
        }
        
        /// <remarks/>
        public Money latePaymentAmount {
            get {
                return this.latePaymentAmountField;
            }
            set {
                this.latePaymentAmountField = value;
            }
        }
        
        /// <remarks/>
        public Money finalPaymentAmount {
            get {
                return this.finalPaymentAmountField;
            }
            set {
                this.finalPaymentAmountField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime maturityDate {
            get {
                return this.maturityDateField;
            }
            set {
                this.maturityDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool maturityDateSpecified {
            get {
                return this.maturityDateFieldSpecified;
            }
            set {
                this.maturityDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="integer")]
        public string paymentDay {
            get {
                return this.paymentDayField;
            }
            set {
                this.paymentDayField = value;
            }
        }
        
        /// <remarks/>
        public LoanPaymentMethodType paymentMethod {
            get {
                return this.paymentMethodField;
            }
            set {
                this.paymentMethodField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool paymentMethodSpecified {
            get {
                return this.paymentMethodFieldSpecified;
            }
            set {
                this.paymentMethodFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string otherPaymentMethod {
            get {
                return this.otherPaymentMethodField;
            }
            set {
                this.otherPaymentMethodField = value;
            }
        }
        
        /// <remarks/>
        public string couponCode {
            get {
                return this.couponCodeField;
            }
            set {
                this.couponCodeField = value;
            }
        }
        
        /// <remarks/>
        public Money amountIncludingPrincipalAndInterest {
            get {
                return this.amountIncludingPrincipalAndInterestField;
            }
            set {
                this.amountIncludingPrincipalAndInterestField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Loan.xsd")]
    public enum PaymentOptionType {
        
        /// <remarks/>
        Fixed,
        
        /// <remarks/>
        LevelPayment,
        
        /// <remarks/>
        LevelPrincipal,
        
        /// <remarks/>
        Variable,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Loan.xsd")]
    public enum LoanPaymentFrequencyType {
        
        /// <remarks/>
        AmortizedSchedule,
        
        /// <remarks/>
        Annual,
        
        /// <remarks/>
        Bimonthly,
        
        /// <remarks/>
        Biweekly,
        
        /// <remarks/>
        BiweeklySkipFirst,
        
        /// <remarks/>
        BiweeklySkipLast,
        
        /// <remarks/>
        Immediate,
        
        /// <remarks/>
        Monthly,
        
        /// <remarks/>
        Quarterly,
        
        /// <remarks/>
        SemiAnnual,
        
        /// <remarks/>
        SemiMonthly,
        
        /// <remarks/>
        SinglePayment,
        
        /// <remarks/>
        Weekly,
        
        /// <remarks/>
        WeeklySkipFirst,
        
        /// <remarks/>
        WeeklySkipLast,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Loan.xsd")]
    public enum LoanPaymentMethodType {
        
        /// <remarks/>
        AutomaticDebit,
        
        /// <remarks/>
        AutomaticTransfer,
        
        /// <remarks/>
        BillPay,
        
        /// <remarks/>
        Cash,
        
        /// <remarks/>
        Check,
        
        /// <remarks/>
        Counter,
        
        /// <remarks/>
        Coupon,
        
        /// <remarks/>
        DebitCard,
        
        /// <remarks/>
        Distribution,
        
        /// <remarks/>
        DistributionAfterDue,
        
        /// <remarks/>
        Electronic,
        
        /// <remarks/>
        Other,
        
        /// <remarks/>
        Payroll,
        
        /// <remarks/>
        PayrollAfterDue,
        
        /// <remarks/>
        Phone,
        
        /// <remarks/>
        ScheduledAutomaticTransfer,
        
        /// <remarks/>
        ScheduledAutomaticTransferAfterDue,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Loan.xsd")]
    public enum LoanInterestCalculationType {
        
        /// <remarks/>
        Actual360,
        
        /// <remarks/>
        Actual364,
        
        /// <remarks/>
        Daily365,
        
        /// <remarks/>
        Daily365Leap,
        
        /// <remarks/>
        DailyBilled360,
        
        /// <remarks/>
        Monthly360,
        
        /// <remarks/>
        Scheduled364,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Loan.xsd")]
    public enum BalloonDueDateTermType {
        
        /// <remarks/>
        Months,
        
        /// <remarks/>
        Years,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Loan.xsd")]
    public enum PrepaymentPenaltyBasisType {
        
        /// <remarks/>
        Amount,
        
        /// <remarks/>
        PercentOfBalance,
        
        /// <remarks/>
        PercentOfOriginalAmount,
        
        /// <remarks/>
        PercentOfPrincipal,
        
        /// <remarks/>
        PercentOfUndisbursed,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Loan.xsd")]
    public partial class LoanFee {
        
        private string loanFeeIdField;
        
        private LateChargeFee lateChargeFeeField;
        
        private Money feeAmountField;
        
        private bool includeEscrowInLateChargeField;
        
        private bool includeEscrowInLateChargeFieldSpecified;
        
        private string feeCodeField;
        
        private Money feeMinField;
        
        private Money feeMaxField;
        
        private Money prePaidAmountField;
        
        private Money financedFeeAmountField;
        
        private Money unamortizedFeesField;
        
        private string feeDescriptionField;
        
        /// <remarks/>
        public string loanFeeId {
            get {
                return this.loanFeeIdField;
            }
            set {
                this.loanFeeIdField = value;
            }
        }
        
        /// <remarks/>
        public LateChargeFee lateChargeFee {
            get {
                return this.lateChargeFeeField;
            }
            set {
                this.lateChargeFeeField = value;
            }
        }
        
        /// <remarks/>
        public Money feeAmount {
            get {
                return this.feeAmountField;
            }
            set {
                this.feeAmountField = value;
            }
        }
        
        /// <remarks/>
        public bool includeEscrowInLateCharge {
            get {
                return this.includeEscrowInLateChargeField;
            }
            set {
                this.includeEscrowInLateChargeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool includeEscrowInLateChargeSpecified {
            get {
                return this.includeEscrowInLateChargeFieldSpecified;
            }
            set {
                this.includeEscrowInLateChargeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string feeCode {
            get {
                return this.feeCodeField;
            }
            set {
                this.feeCodeField = value;
            }
        }
        
        /// <remarks/>
        public Money feeMin {
            get {
                return this.feeMinField;
            }
            set {
                this.feeMinField = value;
            }
        }
        
        /// <remarks/>
        public Money feeMax {
            get {
                return this.feeMaxField;
            }
            set {
                this.feeMaxField = value;
            }
        }
        
        /// <remarks/>
        public Money prePaidAmount {
            get {
                return this.prePaidAmountField;
            }
            set {
                this.prePaidAmountField = value;
            }
        }
        
        /// <remarks/>
        public Money financedFeeAmount {
            get {
                return this.financedFeeAmountField;
            }
            set {
                this.financedFeeAmountField = value;
            }
        }
        
        /// <remarks/>
        public Money unamortizedFees {
            get {
                return this.unamortizedFeesField;
            }
            set {
                this.unamortizedFeesField = value;
            }
        }
        
        /// <remarks/>
        public string feeDescription {
            get {
                return this.feeDescriptionField;
            }
            set {
                this.feeDescriptionField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Loan.xsd")]
    public partial class LateChargeFee {
        
        private string lateChargeFormulaCodeField;
        
        private string numberOfLateChargeGraceDaysField;
        
        private bool addLateChargeToStandardPaymentField;
        
        private bool addLateChargeToStandardPaymentFieldSpecified;
        
        private decimal lateChargePercentageRateField;
        
        private bool lateChargePercentageRateFieldSpecified;
        
        /// <remarks/>
        public string lateChargeFormulaCode {
            get {
                return this.lateChargeFormulaCodeField;
            }
            set {
                this.lateChargeFormulaCodeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="integer")]
        public string numberOfLateChargeGraceDays {
            get {
                return this.numberOfLateChargeGraceDaysField;
            }
            set {
                this.numberOfLateChargeGraceDaysField = value;
            }
        }
        
        /// <remarks/>
        public bool addLateChargeToStandardPayment {
            get {
                return this.addLateChargeToStandardPaymentField;
            }
            set {
                this.addLateChargeToStandardPaymentField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool addLateChargeToStandardPaymentSpecified {
            get {
                return this.addLateChargeToStandardPaymentFieldSpecified;
            }
            set {
                this.addLateChargeToStandardPaymentFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public decimal lateChargePercentageRate {
            get {
                return this.lateChargePercentageRateField;
            }
            set {
                this.lateChargePercentageRateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool lateChargePercentageRateSpecified {
            get {
                return this.lateChargePercentageRateFieldSpecified;
            }
            set {
                this.lateChargePercentageRateFieldSpecified = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Loan.xsd")]
    public partial class InsuranceAddOn {
        
        private string insuranceAddOnIdField;
        
        private LoanInsuranceType insuranceTypeField;
        
        private bool insuranceTypeFieldSpecified;
        
        private string otherInsuranceTypeField;
        
        private Money insuranceMaximumField;
        
        private Money insuranceAmountField;
        
        private Money insuranceFeeAmountField;
        
        private string insuranceSubTypeField;
        
        private string insuranceShortDescriptionField;
        
        private string insuranceDescriptionField;
        
        private bool insurancePostingConsolidationFlagField;
        
        private bool insurancePostingConsolidationFlagFieldSpecified;
        
        private System.DateTime primaryInsuredDateOfBirthField;
        
        private bool primaryInsuredDateOfBirthFieldSpecified;
        
        private System.DateTime secondInsuredDateOfBirthField;
        
        private bool secondInsuredDateOfBirthFieldSpecified;
        
        /// <remarks/>
        public string insuranceAddOnId {
            get {
                return this.insuranceAddOnIdField;
            }
            set {
                this.insuranceAddOnIdField = value;
            }
        }
        
        /// <remarks/>
        public LoanInsuranceType insuranceType {
            get {
                return this.insuranceTypeField;
            }
            set {
                this.insuranceTypeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool insuranceTypeSpecified {
            get {
                return this.insuranceTypeFieldSpecified;
            }
            set {
                this.insuranceTypeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string otherInsuranceType {
            get {
                return this.otherInsuranceTypeField;
            }
            set {
                this.otherInsuranceTypeField = value;
            }
        }
        
        /// <remarks/>
        public Money insuranceMaximum {
            get {
                return this.insuranceMaximumField;
            }
            set {
                this.insuranceMaximumField = value;
            }
        }
        
        /// <remarks/>
        public Money insuranceAmount {
            get {
                return this.insuranceAmountField;
            }
            set {
                this.insuranceAmountField = value;
            }
        }
        
        /// <remarks/>
        public Money insuranceFeeAmount {
            get {
                return this.insuranceFeeAmountField;
            }
            set {
                this.insuranceFeeAmountField = value;
            }
        }
        
        /// <remarks/>
        public string insuranceSubType {
            get {
                return this.insuranceSubTypeField;
            }
            set {
                this.insuranceSubTypeField = value;
            }
        }
        
        /// <remarks/>
        public string insuranceShortDescription {
            get {
                return this.insuranceShortDescriptionField;
            }
            set {
                this.insuranceShortDescriptionField = value;
            }
        }
        
        /// <remarks/>
        public string insuranceDescription {
            get {
                return this.insuranceDescriptionField;
            }
            set {
                this.insuranceDescriptionField = value;
            }
        }
        
        /// <remarks/>
        public bool insurancePostingConsolidationFlag {
            get {
                return this.insurancePostingConsolidationFlagField;
            }
            set {
                this.insurancePostingConsolidationFlagField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool insurancePostingConsolidationFlagSpecified {
            get {
                return this.insurancePostingConsolidationFlagFieldSpecified;
            }
            set {
                this.insurancePostingConsolidationFlagFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime primaryInsuredDateOfBirth {
            get {
                return this.primaryInsuredDateOfBirthField;
            }
            set {
                this.primaryInsuredDateOfBirthField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool primaryInsuredDateOfBirthSpecified {
            get {
                return this.primaryInsuredDateOfBirthFieldSpecified;
            }
            set {
                this.primaryInsuredDateOfBirthFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime secondInsuredDateOfBirth {
            get {
                return this.secondInsuredDateOfBirthField;
            }
            set {
                this.secondInsuredDateOfBirthField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool secondInsuredDateOfBirthSpecified {
            get {
                return this.secondInsuredDateOfBirthFieldSpecified;
            }
            set {
                this.secondInsuredDateOfBirthFieldSpecified = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Loan.xsd")]
    public enum LoanInsuranceType {
        
        /// <remarks/>
        AccidentalDeathAndDismemberment,
        
        /// <remarks/>
        DebtCancellation,
        
        /// <remarks/>
        Disability,
        
        /// <remarks/>
        Gap,
        
        /// <remarks/>
        Life,
        
        /// <remarks/>
        Other,
        
        /// <remarks/>
        PaymentProtection,
        
        /// <remarks/>
        Mbi,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Collateral.xsd")]
    public partial class Collateral {
        
        private CollateralBase itemField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("aircraftCollateral", typeof(AircraftCollateral))]
        [System.Xml.Serialization.XmlElementAttribute("boatCollateral", typeof(BoatCollateral))]
        [System.Xml.Serialization.XmlElementAttribute("letterOfCreditCollateral", typeof(LetterOfCreditCollateral))]
        [System.Xml.Serialization.XmlElementAttribute("mobileHomeCollateral", typeof(MobileHomeCollateral))]
        [System.Xml.Serialization.XmlElementAttribute("motorVehicleCollateral", typeof(MotorVehicleCollateral))]
        [System.Xml.Serialization.XmlElementAttribute("otherTitledCollateral", typeof(OtherTitledCollateral))]
        [System.Xml.Serialization.XmlElementAttribute("realEstateCollateral", typeof(RealEstateCollateral))]
        [System.Xml.Serialization.XmlElementAttribute("savingsCDCollateral", typeof(SavingsCDCollateral))]
        [System.Xml.Serialization.XmlElementAttribute("securitiesCollateral", typeof(SecuritiesCollateral))]
        [System.Xml.Serialization.XmlElementAttribute("shipCollateral", typeof(ShipCollateral))]
        [System.Xml.Serialization.XmlElementAttribute("trailerCollateral", typeof(TrailerCollateral))]
        [System.Xml.Serialization.XmlElementAttribute("uccCollateral", typeof(UccCollateral))]
        public CollateralBase Item {
            get {
                return this.itemField;
            }
            set {
                this.itemField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Collateral.xsd")]
    public partial class AircraftCollateral : TitledCollateralBase {
        
        private string serialNumberField;
        
        private string faaRegistrationNumberField;
        
        private bool inspectionReportYNField;
        
        private bool inspectionReportYNFieldSpecified;
        
        private bool faaCoverLetterYNField;
        
        private bool faaCoverLetterYNFieldSpecified;
        
        private bool aircraftForCommercialUseField;
        
        private bool aircraftForCommercialUseFieldSpecified;
        
        private string descAvionicsEnginesField;
        
        private string descOfLogBooksField;
        
        private string airportHomeBaseField;
        
        private string hoursField;
        
        /// <remarks/>
        public string serialNumber {
            get {
                return this.serialNumberField;
            }
            set {
                this.serialNumberField = value;
            }
        }
        
        /// <remarks/>
        public string faaRegistrationNumber {
            get {
                return this.faaRegistrationNumberField;
            }
            set {
                this.faaRegistrationNumberField = value;
            }
        }
        
        /// <remarks/>
        public bool inspectionReportYN {
            get {
                return this.inspectionReportYNField;
            }
            set {
                this.inspectionReportYNField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool inspectionReportYNSpecified {
            get {
                return this.inspectionReportYNFieldSpecified;
            }
            set {
                this.inspectionReportYNFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public bool faaCoverLetterYN {
            get {
                return this.faaCoverLetterYNField;
            }
            set {
                this.faaCoverLetterYNField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool faaCoverLetterYNSpecified {
            get {
                return this.faaCoverLetterYNFieldSpecified;
            }
            set {
                this.faaCoverLetterYNFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public bool aircraftForCommercialUse {
            get {
                return this.aircraftForCommercialUseField;
            }
            set {
                this.aircraftForCommercialUseField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool aircraftForCommercialUseSpecified {
            get {
                return this.aircraftForCommercialUseFieldSpecified;
            }
            set {
                this.aircraftForCommercialUseFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string descAvionicsEngines {
            get {
                return this.descAvionicsEnginesField;
            }
            set {
                this.descAvionicsEnginesField = value;
            }
        }
        
        /// <remarks/>
        public string descOfLogBooks {
            get {
                return this.descOfLogBooksField;
            }
            set {
                this.descOfLogBooksField = value;
            }
        }
        
        /// <remarks/>
        public string airportHomeBase {
            get {
                return this.airportHomeBaseField;
            }
            set {
                this.airportHomeBaseField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="integer")]
        public string hours {
            get {
                return this.hoursField;
            }
            set {
                this.hoursField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Collateral.xsd")]
    public partial class TitledCollateralBase : CollateralBase {
        
        private string yearField;
        
        private string manufacturerField;
        
        private string makeField;
        
        private string modelField;
        
        private string colorField;
        
        private Money purchasePriceField;
        
        private bool isVehicleUsedField;
        
        private bool isVehicleUsedFieldSpecified;
        
        private string uccCodeField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="integer")]
        public string year {
            get {
                return this.yearField;
            }
            set {
                this.yearField = value;
            }
        }
        
        /// <remarks/>
        public string manufacturer {
            get {
                return this.manufacturerField;
            }
            set {
                this.manufacturerField = value;
            }
        }
        
        /// <remarks/>
        public string make {
            get {
                return this.makeField;
            }
            set {
                this.makeField = value;
            }
        }
        
        /// <remarks/>
        public string model {
            get {
                return this.modelField;
            }
            set {
                this.modelField = value;
            }
        }
        
        /// <remarks/>
        public string color {
            get {
                return this.colorField;
            }
            set {
                this.colorField = value;
            }
        }
        
        /// <remarks/>
        public Money purchasePrice {
            get {
                return this.purchasePriceField;
            }
            set {
                this.purchasePriceField = value;
            }
        }
        
        /// <remarks/>
        public bool isVehicleUsed {
            get {
                return this.isVehicleUsedField;
            }
            set {
                this.isVehicleUsedField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool isVehicleUsedSpecified {
            get {
                return this.isVehicleUsedFieldSpecified;
            }
            set {
                this.isVehicleUsedFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string uccCode {
            get {
                return this.uccCodeField;
            }
            set {
                this.uccCodeField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Collateral.xsd")]
    public partial class CollateralBase {
        
        private string collateralIdField;
        
        private string collateralCodeField;
        
        private System.DateTime collateralPledgedDateField;
        
        private bool collateralPledgedDateFieldSpecified;
        
        private string descriptionField;
        
        private string securedCodeField;
        
        private Money totalSecuredAmountField;
        
        private Money valuationOfCollateralField;
        
        private string ownerOfCollateralField;
        
        private decimal loanToValuePercentageField;
        
        private bool loanToValuePercentageFieldSpecified;
        
        private System.DateTime collateralValueSourceDateField;
        
        private bool collateralValueSourceDateFieldSpecified;
        
        private string collateralValueSourceField;
        
        /// <remarks/>
        public string collateralId {
            get {
                return this.collateralIdField;
            }
            set {
                this.collateralIdField = value;
            }
        }
        
        /// <remarks/>
        public string collateralCode {
            get {
                return this.collateralCodeField;
            }
            set {
                this.collateralCodeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime collateralPledgedDate {
            get {
                return this.collateralPledgedDateField;
            }
            set {
                this.collateralPledgedDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool collateralPledgedDateSpecified {
            get {
                return this.collateralPledgedDateFieldSpecified;
            }
            set {
                this.collateralPledgedDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string description {
            get {
                return this.descriptionField;
            }
            set {
                this.descriptionField = value;
            }
        }
        
        /// <remarks/>
        public string securedCode {
            get {
                return this.securedCodeField;
            }
            set {
                this.securedCodeField = value;
            }
        }
        
        /// <remarks/>
        public Money totalSecuredAmount {
            get {
                return this.totalSecuredAmountField;
            }
            set {
                this.totalSecuredAmountField = value;
            }
        }
        
        /// <remarks/>
        public Money valuationOfCollateral {
            get {
                return this.valuationOfCollateralField;
            }
            set {
                this.valuationOfCollateralField = value;
            }
        }
        
        /// <remarks/>
        public string ownerOfCollateral {
            get {
                return this.ownerOfCollateralField;
            }
            set {
                this.ownerOfCollateralField = value;
            }
        }
        
        /// <remarks/>
        public decimal loanToValuePercentage {
            get {
                return this.loanToValuePercentageField;
            }
            set {
                this.loanToValuePercentageField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool loanToValuePercentageSpecified {
            get {
                return this.loanToValuePercentageFieldSpecified;
            }
            set {
                this.loanToValuePercentageFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime collateralValueSourceDate {
            get {
                return this.collateralValueSourceDateField;
            }
            set {
                this.collateralValueSourceDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool collateralValueSourceDateSpecified {
            get {
                return this.collateralValueSourceDateFieldSpecified;
            }
            set {
                this.collateralValueSourceDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string collateralValueSource {
            get {
                return this.collateralValueSourceField;
            }
            set {
                this.collateralValueSourceField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Collateral.xsd")]
    public abstract partial class PossessoryCollateralBase : CollateralBase {
        
        private Money securityCollateralValueLimitField;
        
        private System.Nullable<decimal> securityMarketValueLimitField;
        
        private bool securityMarketValueLimitFieldSpecified;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(IsNullable=true)]
        public Money securityCollateralValueLimit {
            get {
                return this.securityCollateralValueLimitField;
            }
            set {
                this.securityCollateralValueLimitField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(IsNullable=true)]
        public System.Nullable<decimal> securityMarketValueLimit {
            get {
                return this.securityMarketValueLimitField;
            }
            set {
                this.securityMarketValueLimitField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool securityMarketValueLimitSpecified {
            get {
                return this.securityMarketValueLimitFieldSpecified;
            }
            set {
                this.securityMarketValueLimitFieldSpecified = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Collateral.xsd")]
    public partial class LetterOfCreditCollateral : PossessoryCollateralBase {
        
        private string possessoryNumberField;
        
        private System.Nullable<System.DateTime> issueDateField;
        
        private bool issueDateFieldSpecified;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(IsNullable=true)]
        public string possessoryNumber {
            get {
                return this.possessoryNumberField;
            }
            set {
                this.possessoryNumberField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date", IsNullable=true)]
        public System.Nullable<System.DateTime> issueDate {
            get {
                return this.issueDateField;
            }
            set {
                this.issueDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool issueDateSpecified {
            get {
                return this.issueDateFieldSpecified;
            }
            set {
                this.issueDateFieldSpecified = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Collateral.xsd")]
    public partial class SecuritiesCollateral : PossessoryCollateralBase {
        
        private string possessoryNumberField;
        
        private string faceValueField;
        
        private string issuerField;
        
        private System.Nullable<decimal> numberOfSharesField;
        
        private bool numberOfSharesFieldSpecified;
        
        private string cusipNumberField;
        
        private string heldByField;
        
        private System.Nullable<bool> bookEntryField;
        
        private bool bookEntryFieldSpecified;
        
        private System.Nullable<ValuationFrequency> valuationFrequencyField;
        
        private bool valuationFrequencyFieldSpecified;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(IsNullable=true)]
        public string possessoryNumber {
            get {
                return this.possessoryNumberField;
            }
            set {
                this.possessoryNumberField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(IsNullable=true)]
        public string faceValue {
            get {
                return this.faceValueField;
            }
            set {
                this.faceValueField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(IsNullable=true)]
        public string issuer {
            get {
                return this.issuerField;
            }
            set {
                this.issuerField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(IsNullable=true)]
        public System.Nullable<decimal> numberOfShares {
            get {
                return this.numberOfSharesField;
            }
            set {
                this.numberOfSharesField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool numberOfSharesSpecified {
            get {
                return this.numberOfSharesFieldSpecified;
            }
            set {
                this.numberOfSharesFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(IsNullable=true)]
        public string cusipNumber {
            get {
                return this.cusipNumberField;
            }
            set {
                this.cusipNumberField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(IsNullable=true)]
        public string heldBy {
            get {
                return this.heldByField;
            }
            set {
                this.heldByField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(IsNullable=true)]
        public System.Nullable<bool> bookEntry {
            get {
                return this.bookEntryField;
            }
            set {
                this.bookEntryField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool bookEntrySpecified {
            get {
                return this.bookEntryFieldSpecified;
            }
            set {
                this.bookEntryFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(IsNullable=true)]
        public System.Nullable<ValuationFrequency> valuationFrequency {
            get {
                return this.valuationFrequencyField;
            }
            set {
                this.valuationFrequencyField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool valuationFrequencySpecified {
            get {
                return this.valuationFrequencyFieldSpecified;
            }
            set {
                this.valuationFrequencyFieldSpecified = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Collateral.xsd")]
    public enum ValuationFrequency {
        
        /// <remarks/>
        Weekly,
        
        /// <remarks/>
        BiWeekly,
        
        /// <remarks/>
        Monthly,
        
        /// <remarks/>
        BiMonthly,
        
        /// <remarks/>
        Quarterly,
        
        /// <remarks/>
        SemiAnnually,
        
        /// <remarks/>
        Annually,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Collateral.xsd")]
    public partial class SavingsCDCollateral : PossessoryCollateralBase {
        
        private string accountNumberField;
        
        private string faceValueField;
        
        private string issuerTypeField;
        
        private System.Nullable<bool> certifiedField;
        
        private bool certifiedFieldSpecified;
        
        private Money possessoryInterestRateField;
        
        private Money amountField;
        
        private decimal percentageField;
        
        private bool percentageFieldSpecified;
        
        /// <remarks/>
        public string accountNumber {
            get {
                return this.accountNumberField;
            }
            set {
                this.accountNumberField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(IsNullable=true)]
        public string faceValue {
            get {
                return this.faceValueField;
            }
            set {
                this.faceValueField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(IsNullable=true)]
        public string issuerType {
            get {
                return this.issuerTypeField;
            }
            set {
                this.issuerTypeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(IsNullable=true)]
        public System.Nullable<bool> certified {
            get {
                return this.certifiedField;
            }
            set {
                this.certifiedField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool certifiedSpecified {
            get {
                return this.certifiedFieldSpecified;
            }
            set {
                this.certifiedFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(IsNullable=true)]
        public Money possessoryInterestRate {
            get {
                return this.possessoryInterestRateField;
            }
            set {
                this.possessoryInterestRateField = value;
            }
        }
        
        /// <remarks/>
        public Money amount {
            get {
                return this.amountField;
            }
            set {
                this.amountField = value;
            }
        }
        
        /// <remarks/>
        public decimal percentage {
            get {
                return this.percentageField;
            }
            set {
                this.percentageField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool percentageSpecified {
            get {
                return this.percentageFieldSpecified;
            }
            set {
                this.percentageFieldSpecified = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Collateral.xsd")]
    public partial class UccCollateral : CollateralBase {
        
        private System.Nullable<bool> financingStatementField;
        
        private bool financingStatementFieldSpecified;
        
        private string generalDescriptionField;
        
        private string tortClaimDescriptionField;
        
        private string immovableDescriptionField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(IsNullable=true)]
        public System.Nullable<bool> financingStatement {
            get {
                return this.financingStatementField;
            }
            set {
                this.financingStatementField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool financingStatementSpecified {
            get {
                return this.financingStatementFieldSpecified;
            }
            set {
                this.financingStatementFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(IsNullable=true)]
        public string generalDescription {
            get {
                return this.generalDescriptionField;
            }
            set {
                this.generalDescriptionField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(IsNullable=true)]
        public string tortClaimDescription {
            get {
                return this.tortClaimDescriptionField;
            }
            set {
                this.tortClaimDescriptionField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(IsNullable=true)]
        public string immovableDescription {
            get {
                return this.immovableDescriptionField;
            }
            set {
                this.immovableDescriptionField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Collateral.xsd")]
    public partial class RealEstateCollateral : CollateralBase {
        
        private string taxIdField;
        
        private string parcelIDNNumberField;
        
        private string sectionField;
        
        private string blockField;
        
        private string lotField;
        
        private System.Nullable<bool> firstLienField;
        
        private bool firstLienFieldSpecified;
        
        private System.Nullable<bool> existingLiensField;
        
        private bool existingLiensFieldSpecified;
        
        private System.Nullable<bool> dwellingField;
        
        private bool dwellingFieldSpecified;
        
        private System.Nullable<float> numberOfUnitsField;
        
        private bool numberOfUnitsFieldSpecified;
        
        private System.Nullable<bool> ownerOccupiedField;
        
        private bool ownerOccupiedFieldSpecified;
        
        private System.Nullable<bool> principalDwellingField;
        
        private bool principalDwellingFieldSpecified;
        
        private System.Nullable<bool> assumableField;
        
        private bool assumableFieldSpecified;
        
        private System.Nullable<bool> manufacturedHousingField;
        
        private bool manufacturedHousingFieldSpecified;
        
        private System.Nullable<bool> constructionField;
        
        private bool constructionFieldSpecified;
        
        private System.Nullable<float> yearBuiltField;
        
        private bool yearBuiltFieldSpecified;
        
        private System.Nullable<bool> condoPUDField;
        
        private bool condoPUDFieldSpecified;
        
        private string legalDescriptionField;
        
        private System.Nullable<bool> rentalPropertyField;
        
        private bool rentalPropertyFieldSpecified;
        
        private string yearAcquiredField;
        
        private Money originalCostField;
        
        private Money presentValueField;
        
        private Money improvementsField;
        
        private System.Nullable<bool> timeshareField;
        
        private bool timeshareFieldSpecified;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(IsNullable=true)]
        public string taxId {
            get {
                return this.taxIdField;
            }
            set {
                this.taxIdField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(IsNullable=true)]
        public string parcelIDNNumber {
            get {
                return this.parcelIDNNumberField;
            }
            set {
                this.parcelIDNNumberField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(IsNullable=true)]
        public string section {
            get {
                return this.sectionField;
            }
            set {
                this.sectionField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(IsNullable=true)]
        public string block {
            get {
                return this.blockField;
            }
            set {
                this.blockField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(IsNullable=true)]
        public string lot {
            get {
                return this.lotField;
            }
            set {
                this.lotField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(IsNullable=true)]
        public System.Nullable<bool> firstLien {
            get {
                return this.firstLienField;
            }
            set {
                this.firstLienField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool firstLienSpecified {
            get {
                return this.firstLienFieldSpecified;
            }
            set {
                this.firstLienFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(IsNullable=true)]
        public System.Nullable<bool> existingLiens {
            get {
                return this.existingLiensField;
            }
            set {
                this.existingLiensField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool existingLiensSpecified {
            get {
                return this.existingLiensFieldSpecified;
            }
            set {
                this.existingLiensFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(IsNullable=true)]
        public System.Nullable<bool> dwelling {
            get {
                return this.dwellingField;
            }
            set {
                this.dwellingField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool dwellingSpecified {
            get {
                return this.dwellingFieldSpecified;
            }
            set {
                this.dwellingFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(IsNullable=true)]
        public System.Nullable<float> numberOfUnits {
            get {
                return this.numberOfUnitsField;
            }
            set {
                this.numberOfUnitsField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool numberOfUnitsSpecified {
            get {
                return this.numberOfUnitsFieldSpecified;
            }
            set {
                this.numberOfUnitsFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(IsNullable=true)]
        public System.Nullable<bool> ownerOccupied {
            get {
                return this.ownerOccupiedField;
            }
            set {
                this.ownerOccupiedField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool ownerOccupiedSpecified {
            get {
                return this.ownerOccupiedFieldSpecified;
            }
            set {
                this.ownerOccupiedFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(IsNullable=true)]
        public System.Nullable<bool> principalDwelling {
            get {
                return this.principalDwellingField;
            }
            set {
                this.principalDwellingField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool principalDwellingSpecified {
            get {
                return this.principalDwellingFieldSpecified;
            }
            set {
                this.principalDwellingFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(IsNullable=true)]
        public System.Nullable<bool> assumable {
            get {
                return this.assumableField;
            }
            set {
                this.assumableField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool assumableSpecified {
            get {
                return this.assumableFieldSpecified;
            }
            set {
                this.assumableFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(IsNullable=true)]
        public System.Nullable<bool> manufacturedHousing {
            get {
                return this.manufacturedHousingField;
            }
            set {
                this.manufacturedHousingField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool manufacturedHousingSpecified {
            get {
                return this.manufacturedHousingFieldSpecified;
            }
            set {
                this.manufacturedHousingFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(IsNullable=true)]
        public System.Nullable<bool> construction {
            get {
                return this.constructionField;
            }
            set {
                this.constructionField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool constructionSpecified {
            get {
                return this.constructionFieldSpecified;
            }
            set {
                this.constructionFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(IsNullable=true)]
        public System.Nullable<float> yearBuilt {
            get {
                return this.yearBuiltField;
            }
            set {
                this.yearBuiltField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool yearBuiltSpecified {
            get {
                return this.yearBuiltFieldSpecified;
            }
            set {
                this.yearBuiltFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(IsNullable=true)]
        public System.Nullable<bool> condoPUD {
            get {
                return this.condoPUDField;
            }
            set {
                this.condoPUDField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool condoPUDSpecified {
            get {
                return this.condoPUDFieldSpecified;
            }
            set {
                this.condoPUDFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(IsNullable=true)]
        public string legalDescription {
            get {
                return this.legalDescriptionField;
            }
            set {
                this.legalDescriptionField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(IsNullable=true)]
        public System.Nullable<bool> rentalProperty {
            get {
                return this.rentalPropertyField;
            }
            set {
                this.rentalPropertyField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool rentalPropertySpecified {
            get {
                return this.rentalPropertyFieldSpecified;
            }
            set {
                this.rentalPropertyFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(IsNullable=true)]
        public string yearAcquired {
            get {
                return this.yearAcquiredField;
            }
            set {
                this.yearAcquiredField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(IsNullable=true)]
        public Money originalCost {
            get {
                return this.originalCostField;
            }
            set {
                this.originalCostField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(IsNullable=true)]
        public Money presentValue {
            get {
                return this.presentValueField;
            }
            set {
                this.presentValueField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(IsNullable=true)]
        public Money improvements {
            get {
                return this.improvementsField;
            }
            set {
                this.improvementsField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(IsNullable=true)]
        public System.Nullable<bool> timeshare {
            get {
                return this.timeshareField;
            }
            set {
                this.timeshareField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool timeshareSpecified {
            get {
                return this.timeshareFieldSpecified;
            }
            set {
                this.timeshareFieldSpecified = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Collateral.xsd")]
    public partial class MobileHomeCollateral : TitledCollateralBase {
        
        private string serialNumberField;
        
        private string bodyStyleField;
        
        private string descOfEquipmentField;
        
        /// <remarks/>
        public string serialNumber {
            get {
                return this.serialNumberField;
            }
            set {
                this.serialNumberField = value;
            }
        }
        
        /// <remarks/>
        public string bodyStyle {
            get {
                return this.bodyStyleField;
            }
            set {
                this.bodyStyleField = value;
            }
        }
        
        /// <remarks/>
        public string descOfEquipment {
            get {
                return this.descOfEquipmentField;
            }
            set {
                this.descOfEquipmentField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Collateral.xsd")]
    public partial class OtherTitledCollateral : TitledCollateralBase {
        
        private string bodyStyleField;
        
        private string serialNumberField;
        
        /// <remarks/>
        public string bodyStyle {
            get {
                return this.bodyStyleField;
            }
            set {
                this.bodyStyleField = value;
            }
        }
        
        /// <remarks/>
        public string serialNumber {
            get {
                return this.serialNumberField;
            }
            set {
                this.serialNumberField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Collateral.xsd")]
    public partial class BoatCollateral : TitledCollateralBase {
        
        private string bodyStyleField;
        
        private string serialNumberField;
        
        private string marinaPortField;
        
        private string descOfEnginesEquipField;
        
        private string descOfLogBooksField;
        
        /// <remarks/>
        public string bodyStyle {
            get {
                return this.bodyStyleField;
            }
            set {
                this.bodyStyleField = value;
            }
        }
        
        /// <remarks/>
        public string serialNumber {
            get {
                return this.serialNumberField;
            }
            set {
                this.serialNumberField = value;
            }
        }
        
        /// <remarks/>
        public string marinaPort {
            get {
                return this.marinaPortField;
            }
            set {
                this.marinaPortField = value;
            }
        }
        
        /// <remarks/>
        public string descOfEnginesEquip {
            get {
                return this.descOfEnginesEquipField;
            }
            set {
                this.descOfEnginesEquipField = value;
            }
        }
        
        /// <remarks/>
        public string descOfLogBooks {
            get {
                return this.descOfLogBooksField;
            }
            set {
                this.descOfLogBooksField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Collateral.xsd")]
    public partial class TrailerCollateral : TitledCollateralBase {
        
        private string serialNumberField;
        
        /// <remarks/>
        public string serialNumber {
            get {
                return this.serialNumberField;
            }
            set {
                this.serialNumberField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Collateral.xsd")]
    public partial class ShipCollateral : TitledCollateralBase {
        
        private string nameOfVesselField;
        
        private string officialNumberField;
        
        private string marinaPortField;
        
        private string grossTonnageField;
        
        private string netTonnageField;
        
        /// <remarks/>
        public string nameOfVessel {
            get {
                return this.nameOfVesselField;
            }
            set {
                this.nameOfVesselField = value;
            }
        }
        
        /// <remarks/>
        public string officialNumber {
            get {
                return this.officialNumberField;
            }
            set {
                this.officialNumberField = value;
            }
        }
        
        /// <remarks/>
        public string marinaPort {
            get {
                return this.marinaPortField;
            }
            set {
                this.marinaPortField = value;
            }
        }
        
        /// <remarks/>
        public string grossTonnage {
            get {
                return this.grossTonnageField;
            }
            set {
                this.grossTonnageField = value;
            }
        }
        
        /// <remarks/>
        public string netTonnage {
            get {
                return this.netTonnageField;
            }
            set {
                this.netTonnageField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Collateral.xsd")]
    public partial class MotorVehicleCollateral : TitledCollateralBase {
        
        private string vehicleIdentificationNumberField;
        
        private string mileageField;
        
        private string licensePlateNumberField;
        
        private string licensePlateStateField;
        
        private System.DateTime licensePlateExpirationField;
        
        private bool licensePlateExpirationFieldSpecified;
        
        private string titleNumberField;
        
        private string titleStateField;
        
        /// <remarks/>
        public string vehicleIdentificationNumber {
            get {
                return this.vehicleIdentificationNumberField;
            }
            set {
                this.vehicleIdentificationNumberField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="integer")]
        public string mileage {
            get {
                return this.mileageField;
            }
            set {
                this.mileageField = value;
            }
        }
        
        /// <remarks/>
        public string licensePlateNumber {
            get {
                return this.licensePlateNumberField;
            }
            set {
                this.licensePlateNumberField = value;
            }
        }
        
        /// <remarks/>
        public string licensePlateState {
            get {
                return this.licensePlateStateField;
            }
            set {
                this.licensePlateStateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime licensePlateExpiration {
            get {
                return this.licensePlateExpirationField;
            }
            set {
                this.licensePlateExpirationField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool licensePlateExpirationSpecified {
            get {
                return this.licensePlateExpirationFieldSpecified;
            }
            set {
                this.licensePlateExpirationFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string titleNumber {
            get {
                return this.titleNumberField;
            }
            set {
                this.titleNumberField = value;
            }
        }
        
        /// <remarks/>
        public string titleState {
            get {
                return this.titleStateField;
            }
            set {
                this.titleStateField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Loan.xsd")]
    public partial class AutoPaymentOption {
        
        private AutoPaymentOptionType autoPaymentOptionField;
        
        private bool autoPaymentOptionFieldSpecified;
        
        private string otherAutoPaymentOptionField;
        
        private AutoPaymentFrequencyType autoPaymentFrequencyField;
        
        private bool autoPaymentFrequencyFieldSpecified;
        
        private DayOfTheWeek autoPaymentDayOfTheWeekField;
        
        private bool autoPaymentDayOfTheWeekFieldSpecified;
        
        private string autoPaymentDayOfTheMonthField;
        
        private Money autoPaymentAmountField;
        
        private string fromAccountIdField;
        
        /// <remarks/>
        public AutoPaymentOptionType autoPaymentOption {
            get {
                return this.autoPaymentOptionField;
            }
            set {
                this.autoPaymentOptionField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool autoPaymentOptionSpecified {
            get {
                return this.autoPaymentOptionFieldSpecified;
            }
            set {
                this.autoPaymentOptionFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string otherAutoPaymentOption {
            get {
                return this.otherAutoPaymentOptionField;
            }
            set {
                this.otherAutoPaymentOptionField = value;
            }
        }
        
        /// <remarks/>
        public AutoPaymentFrequencyType autoPaymentFrequency {
            get {
                return this.autoPaymentFrequencyField;
            }
            set {
                this.autoPaymentFrequencyField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool autoPaymentFrequencySpecified {
            get {
                return this.autoPaymentFrequencyFieldSpecified;
            }
            set {
                this.autoPaymentFrequencyFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public DayOfTheWeek autoPaymentDayOfTheWeek {
            get {
                return this.autoPaymentDayOfTheWeekField;
            }
            set {
                this.autoPaymentDayOfTheWeekField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool autoPaymentDayOfTheWeekSpecified {
            get {
                return this.autoPaymentDayOfTheWeekFieldSpecified;
            }
            set {
                this.autoPaymentDayOfTheWeekFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="integer")]
        public string autoPaymentDayOfTheMonth {
            get {
                return this.autoPaymentDayOfTheMonthField;
            }
            set {
                this.autoPaymentDayOfTheMonthField = value;
            }
        }
        
        /// <remarks/>
        public Money autoPaymentAmount {
            get {
                return this.autoPaymentAmountField;
            }
            set {
                this.autoPaymentAmountField = value;
            }
        }
        
        /// <remarks/>
        public string fromAccountId {
            get {
                return this.fromAccountIdField;
            }
            set {
                this.fromAccountIdField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Loan.xsd")]
    public enum AutoPaymentOptionType {
        
        /// <remarks/>
        Balance,
        
        /// <remarks/>
        MinimumDue,
        
        /// <remarks/>
        Other,
        
        /// <remarks/>
        Principal,
        
        /// <remarks/>
        SetPayment,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Loan.xsd")]
    public enum AutoPaymentFrequencyType {
        
        /// <remarks/>
        Bimonthly,
        
        /// <remarks/>
        Biweekly,
        
        /// <remarks/>
        Monthly,
        
        /// <remarks/>
        OneTime,
        
        /// <remarks/>
        Other,
        
        /// <remarks/>
        Weekly,
        
        /// <remarks/>
        Yearly,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Common.xsd")]
    public enum DayOfTheWeek {
        
        /// <remarks/>
        Monday,
        
        /// <remarks/>
        Tuesday,
        
        /// <remarks/>
        Wednesday,
        
        /// <remarks/>
        Thursday,
        
        /// <remarks/>
        Friday,
        
        /// <remarks/>
        Saturday,
        
        /// <remarks/>
        Sunday,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Loan.xsd")]
    public partial class DelinquencyNotice {
        
        private string delinquencyNoticeIdField;
        
        private string codeField;
        
        private string delinquencyNoticeDaysField;
        
        private string delinquencyLockDaysField;
        
        private string descriptionField;
        
        /// <remarks/>
        public string delinquencyNoticeId {
            get {
                return this.delinquencyNoticeIdField;
            }
            set {
                this.delinquencyNoticeIdField = value;
            }
        }
        
        /// <remarks/>
        public string code {
            get {
                return this.codeField;
            }
            set {
                this.codeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="integer")]
        public string delinquencyNoticeDays {
            get {
                return this.delinquencyNoticeDaysField;
            }
            set {
                this.delinquencyNoticeDaysField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="integer")]
        public string delinquencyLockDays {
            get {
                return this.delinquencyLockDaysField;
            }
            set {
                this.delinquencyLockDaysField = value;
            }
        }
        
        /// <remarks/>
        public string description {
            get {
                return this.descriptionField;
            }
            set {
                this.descriptionField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Loan.xsd")]
    public partial class SkipPayment {
        
        private string numberOfSkipPaymentsAllowedField;
        
        private System.DateTime skipPaymentBeginDateField;
        
        private bool skipPaymentBeginDateFieldSpecified;
        
        private MonthType monthToStartSkipPaymentsField;
        
        private bool monthToStartSkipPaymentsFieldSpecified;
        
        private string yearToSkipPaymentField;
        
        private bool isSkipRecurringField;
        
        private bool isSkipRecurringFieldSpecified;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="integer")]
        public string numberOfSkipPaymentsAllowed {
            get {
                return this.numberOfSkipPaymentsAllowedField;
            }
            set {
                this.numberOfSkipPaymentsAllowedField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime skipPaymentBeginDate {
            get {
                return this.skipPaymentBeginDateField;
            }
            set {
                this.skipPaymentBeginDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool skipPaymentBeginDateSpecified {
            get {
                return this.skipPaymentBeginDateFieldSpecified;
            }
            set {
                this.skipPaymentBeginDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public MonthType monthToStartSkipPayments {
            get {
                return this.monthToStartSkipPaymentsField;
            }
            set {
                this.monthToStartSkipPaymentsField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool monthToStartSkipPaymentsSpecified {
            get {
                return this.monthToStartSkipPaymentsFieldSpecified;
            }
            set {
                this.monthToStartSkipPaymentsFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="integer")]
        public string yearToSkipPayment {
            get {
                return this.yearToSkipPaymentField;
            }
            set {
                this.yearToSkipPaymentField = value;
            }
        }
        
        /// <remarks/>
        public bool isSkipRecurring {
            get {
                return this.isSkipRecurringField;
            }
            set {
                this.isSkipRecurringField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool isSkipRecurringSpecified {
            get {
                return this.isSkipRecurringFieldSpecified;
            }
            set {
                this.isSkipRecurringFieldSpecified = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Loan.xsd")]
    public enum MonthType {
        
        /// <remarks/>
        January,
        
        /// <remarks/>
        February,
        
        /// <remarks/>
        March,
        
        /// <remarks/>
        April,
        
        /// <remarks/>
        May,
        
        /// <remarks/>
        June,
        
        /// <remarks/>
        July,
        
        /// <remarks/>
        August,
        
        /// <remarks/>
        September,
        
        /// <remarks/>
        October,
        
        /// <remarks/>
        November,
        
        /// <remarks/>
        December,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Loan.xsd")]
    public partial class CreditLimitIncreaseRequestList {
        
        private System.DateTime requestDateField;
        
        private bool requestDateFieldSpecified;
        
        private System.DateTime limitIncreaseDateField;
        
        private bool limitIncreaseDateFieldSpecified;
        
        private decimal newLimitRequestField;
        
        private bool newLimitRequestFieldSpecified;
        
        private string memoField;
        
        /// <remarks/>
        public System.DateTime requestDate {
            get {
                return this.requestDateField;
            }
            set {
                this.requestDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool requestDateSpecified {
            get {
                return this.requestDateFieldSpecified;
            }
            set {
                this.requestDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime limitIncreaseDate {
            get {
                return this.limitIncreaseDateField;
            }
            set {
                this.limitIncreaseDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool limitIncreaseDateSpecified {
            get {
                return this.limitIncreaseDateFieldSpecified;
            }
            set {
                this.limitIncreaseDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public decimal newLimitRequest {
            get {
                return this.newLimitRequestField;
            }
            set {
                this.newLimitRequestField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool newLimitRequestSpecified {
            get {
                return this.newLimitRequestFieldSpecified;
            }
            set {
                this.newLimitRequestFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string memo {
            get {
                return this.memoField;
            }
            set {
                this.memoField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Deposit.xsd")]
    public partial class Deposit : Account {
        
        private DepositParty[] depositPartyListField;
        
        private System.DateTime bumpEffectiveDateField;
        
        private bool bumpEffectiveDateFieldSpecified;
        
        private System.DateTime bumpExpirationDateField;
        
        private bool bumpExpirationDateFieldSpecified;
        
        private decimal bumpRateField;
        
        private bool bumpRateFieldSpecified;
        
        private DividendPostCodeType dividendPostCodeField;
        
        private bool dividendPostCodeFieldSpecified;
        
        private decimal dividendRateField;
        
        private bool dividendRateFieldSpecified;
        
        private string dividendTransferAccountIdField;
        
        private string dividendTypeField;
        
        private string irsCodeField;
        
        private MaturityPostCodeType maturityPostCodeField;
        
        private bool maturityPostCodeFieldSpecified;
        
        private string maturityTransferAccountIdField;
        
        private Money minimumDepositField;
        
        private Money minimumWithdrawalField;
        
        private Money overdraftToleranceField;
        
        private DepositAccountStatus depositAccountStatusField;
        
        private bool depositAccountStatusFieldSpecified;
        
        private string depositAccountSubStatusField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("depositParty", IsNullable=false)]
        public DepositParty[] depositPartyList {
            get {
                return this.depositPartyListField;
            }
            set {
                this.depositPartyListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime bumpEffectiveDate {
            get {
                return this.bumpEffectiveDateField;
            }
            set {
                this.bumpEffectiveDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool bumpEffectiveDateSpecified {
            get {
                return this.bumpEffectiveDateFieldSpecified;
            }
            set {
                this.bumpEffectiveDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime bumpExpirationDate {
            get {
                return this.bumpExpirationDateField;
            }
            set {
                this.bumpExpirationDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool bumpExpirationDateSpecified {
            get {
                return this.bumpExpirationDateFieldSpecified;
            }
            set {
                this.bumpExpirationDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public decimal bumpRate {
            get {
                return this.bumpRateField;
            }
            set {
                this.bumpRateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool bumpRateSpecified {
            get {
                return this.bumpRateFieldSpecified;
            }
            set {
                this.bumpRateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public DividendPostCodeType dividendPostCode {
            get {
                return this.dividendPostCodeField;
            }
            set {
                this.dividendPostCodeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool dividendPostCodeSpecified {
            get {
                return this.dividendPostCodeFieldSpecified;
            }
            set {
                this.dividendPostCodeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public decimal dividendRate {
            get {
                return this.dividendRateField;
            }
            set {
                this.dividendRateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool dividendRateSpecified {
            get {
                return this.dividendRateFieldSpecified;
            }
            set {
                this.dividendRateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string dividendTransferAccountId {
            get {
                return this.dividendTransferAccountIdField;
            }
            set {
                this.dividendTransferAccountIdField = value;
            }
        }
        
        /// <remarks/>
        public string dividendType {
            get {
                return this.dividendTypeField;
            }
            set {
                this.dividendTypeField = value;
            }
        }
        
        /// <remarks/>
        public string irsCode {
            get {
                return this.irsCodeField;
            }
            set {
                this.irsCodeField = value;
            }
        }
        
        /// <remarks/>
        public MaturityPostCodeType maturityPostCode {
            get {
                return this.maturityPostCodeField;
            }
            set {
                this.maturityPostCodeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool maturityPostCodeSpecified {
            get {
                return this.maturityPostCodeFieldSpecified;
            }
            set {
                this.maturityPostCodeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string maturityTransferAccountId {
            get {
                return this.maturityTransferAccountIdField;
            }
            set {
                this.maturityTransferAccountIdField = value;
            }
        }
        
        /// <remarks/>
        public Money minimumDeposit {
            get {
                return this.minimumDepositField;
            }
            set {
                this.minimumDepositField = value;
            }
        }
        
        /// <remarks/>
        public Money minimumWithdrawal {
            get {
                return this.minimumWithdrawalField;
            }
            set {
                this.minimumWithdrawalField = value;
            }
        }
        
        /// <remarks/>
        public Money overdraftTolerance {
            get {
                return this.overdraftToleranceField;
            }
            set {
                this.overdraftToleranceField = value;
            }
        }
        
        /// <remarks/>
        public DepositAccountStatus depositAccountStatus {
            get {
                return this.depositAccountStatusField;
            }
            set {
                this.depositAccountStatusField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool depositAccountStatusSpecified {
            get {
                return this.depositAccountStatusFieldSpecified;
            }
            set {
                this.depositAccountStatusFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string depositAccountSubStatus {
            get {
                return this.depositAccountSubStatusField;
            }
            set {
                this.depositAccountSubStatusField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Deposit.xsd")]
    public partial class DepositParty {
        
        private string depositPartyIdField;
        
        private DepositPartyRelationshipType depositPartyRelationshipTypeField;
        
        private bool ssnOverrideField;
        
        private string[] contactIdListField;
        
        public DepositParty() {
            this.ssnOverrideField = false;
        }
        
        /// <remarks/>
        public string depositPartyId {
            get {
                return this.depositPartyIdField;
            }
            set {
                this.depositPartyIdField = value;
            }
        }
        
        /// <remarks/>
        public DepositPartyRelationshipType depositPartyRelationshipType {
            get {
                return this.depositPartyRelationshipTypeField;
            }
            set {
                this.depositPartyRelationshipTypeField = value;
            }
        }
        
        /// <remarks/>
        [System.ComponentModel.DefaultValueAttribute(false)]
        public bool ssnOverride {
            get {
                return this.ssnOverrideField;
            }
            set {
                this.ssnOverrideField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("contactId", Namespace="http://cufxstandards.com/v3/Contact.xsd", IsNullable=false)]
        public string[] contactIdList {
            get {
                return this.contactIdListField;
            }
            set {
                this.contactIdListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Deposit.xsd")]
    public partial class DepositPartyRelationshipType {
        
        private object itemField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("agent", typeof(Agent))]
        [System.Xml.Serialization.XmlElementAttribute("beneficiary", typeof(Beneficiary))]
        [System.Xml.Serialization.XmlElementAttribute("guarantor", typeof(Guarantor))]
        [System.Xml.Serialization.XmlElementAttribute("holder", typeof(Holder))]
        [System.Xml.Serialization.XmlElementAttribute("safeDepositBoxUser", typeof(SafeDepositBoxUser))]
        public object Item {
            get {
                return this.itemField;
            }
            set {
                this.itemField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Relationship.xsd")]
    public partial class Beneficiary {
        
        private BeneficiaryQualifier qualifierField;
        
        private Authority authorityField;
        
        private decimal beneficiaryPercentField;
        
        private BeneficiaryTypes beneficiaryTypesField;
        
        public Beneficiary() {
            this.authorityField = Authority.Unauthorized;
            this.beneficiaryPercentField = ((decimal)(1.00m));
        }
        
        /// <remarks/>
        public BeneficiaryQualifier qualifier {
            get {
                return this.qualifierField;
            }
            set {
                this.qualifierField = value;
            }
        }
        
        /// <remarks/>
        public Authority authority {
            get {
                return this.authorityField;
            }
            set {
                this.authorityField = value;
            }
        }
        
        /// <remarks/>
        public decimal beneficiaryPercent {
            get {
                return this.beneficiaryPercentField;
            }
            set {
                this.beneficiaryPercentField = value;
            }
        }
        
        /// <remarks/>
        public BeneficiaryTypes beneficiaryTypes {
            get {
                return this.beneficiaryTypesField;
            }
            set {
                this.beneficiaryTypesField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Relationship.xsd")]
    public enum BeneficiaryQualifier {
        
        /// <remarks/>
        Standard,
        
        /// <remarks/>
        Education,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Relationship.xsd")]
    public enum BeneficiaryTypes {
        
        /// <remarks/>
        Primary,
        
        /// <remarks/>
        Contingent,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Relationship.xsd")]
    public partial class Holder {
        
        private PrimaryJoint qualifierField;
        
        private Authority authorityField;
        
        /// <remarks/>
        public PrimaryJoint qualifier {
            get {
                return this.qualifierField;
            }
            set {
                this.qualifierField = value;
            }
        }
        
        /// <remarks/>
        public Authority authority {
            get {
                return this.authorityField;
            }
            set {
                this.authorityField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Relationship.xsd")]
    public partial class SafeDepositBoxUser {
        
        private SafeDepositBoxUserQualifier qualifierField;
        
        private Authority authorityField;
        
        public SafeDepositBoxUser() {
            this.authorityField = Authority.Unauthorized;
        }
        
        /// <remarks/>
        public SafeDepositBoxUserQualifier qualifier {
            get {
                return this.qualifierField;
            }
            set {
                this.qualifierField = value;
            }
        }
        
        /// <remarks/>
        public Authority authority {
            get {
                return this.authorityField;
            }
            set {
                this.authorityField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Relationship.xsd")]
    public enum SafeDepositBoxUserQualifier {
        
        /// <remarks/>
        Colessee,
        
        /// <remarks/>
        Deputy,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Deposit.xsd")]
    public enum DividendPostCodeType {
        
        /// <remarks/>
        ToAccount,
        
        /// <remarks/>
        ByCheck,
        
        /// <remarks/>
        Transfer,
        
        /// <remarks/>
        Forfeit,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Deposit.xsd")]
    public enum MaturityPostCodeType {
        
        /// <remarks/>
        Renew,
        
        /// <remarks/>
        ByCheck,
        
        /// <remarks/>
        Transfer,
        
        /// <remarks/>
        Suspend,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Deposit.xsd")]
    public enum DepositAccountStatus {
        
        /// <remarks/>
        Active,
        
        /// <remarks/>
        Closed,
        
        /// <remarks/>
        Dormant,
        
        /// <remarks/>
        Escheated,
        
        /// <remarks/>
        Incomplete,
        
        /// <remarks/>
        Locked,
        
        /// <remarks/>
        Matured,
        
        /// <remarks/>
        RenewPending,
        
        /// <remarks/>
        Restricted,
        
        /// <remarks/>
        Unfunded,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/AccountFilter.xsd")]
    public partial class AccountFilter {
        
        private string[] accountIdListField;
        
        private string[] partyIdListField;
        
        private string[] relationshipIdListField;
        
        private AccountType[] accountTypeListField;
        
        private bool externalAccountFlagField;
        
        private bool externalAccountFlagFieldSpecified;
        
        private bool includeNotesFlagField;
        
        private bool includeNotesFlagFieldSpecified;
        
        private System.DateTime transactionStartDateTimeField;
        
        private bool transactionStartDateTimeFieldSpecified;
        
        private System.DateTime transactionEndDateTimeField;
        
        private bool transactionEndDateTimeFieldSpecified;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("accountId", Namespace="http://cufxstandards.com/v3/Account.xsd", IsNullable=false)]
        public string[] accountIdList {
            get {
                return this.accountIdListField;
            }
            set {
                this.accountIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("partyId", Namespace="http://cufxstandards.com/v3/Party.xsd", IsNullable=false)]
        public string[] partyIdList {
            get {
                return this.partyIdListField;
            }
            set {
                this.partyIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("relationshipId", Namespace="http://cufxstandards.com/v3/Relationship.xsd", IsNullable=false)]
        public string[] relationshipIdList {
            get {
                return this.relationshipIdListField;
            }
            set {
                this.relationshipIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("accountType", Namespace="http://cufxstandards.com/v3/Account.xsd", IsNullable=false)]
        public AccountType[] accountTypeList {
            get {
                return this.accountTypeListField;
            }
            set {
                this.accountTypeListField = value;
            }
        }
        
        /// <remarks/>
        public bool externalAccountFlag {
            get {
                return this.externalAccountFlagField;
            }
            set {
                this.externalAccountFlagField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool externalAccountFlagSpecified {
            get {
                return this.externalAccountFlagFieldSpecified;
            }
            set {
                this.externalAccountFlagFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public bool includeNotesFlag {
            get {
                return this.includeNotesFlagField;
            }
            set {
                this.includeNotesFlagField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool includeNotesFlagSpecified {
            get {
                return this.includeNotesFlagFieldSpecified;
            }
            set {
                this.includeNotesFlagFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime transactionStartDateTime {
            get {
                return this.transactionStartDateTimeField;
            }
            set {
                this.transactionStartDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool transactionStartDateTimeSpecified {
            get {
                return this.transactionStartDateTimeFieldSpecified;
            }
            set {
                this.transactionStartDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime transactionEndDateTime {
            get {
                return this.transactionEndDateTimeField;
            }
            set {
                this.transactionEndDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool transactionEndDateTimeSpecified {
            get {
                return this.transactionEndDateTimeFieldSpecified;
            }
            set {
                this.transactionEndDateTimeFieldSpecified = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/LoanFilter.xsd")]
    public partial class LoanFilter : AccountFilter {
        
        private LoanAccountCategory[] loanCategoryListField;
        
        private LoanAccountStatus[] loanAccountStatusListField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("loanCategory", Namespace="http://cufxstandards.com/v3/Loan.xsd", IsNullable=false)]
        public LoanAccountCategory[] loanCategoryList {
            get {
                return this.loanCategoryListField;
            }
            set {
                this.loanCategoryListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("loanAccountStatus", Namespace="http://cufxstandards.com/v3/Loan.xsd", IsNullable=false)]
        public LoanAccountStatus[] loanAccountStatusList {
            get {
                return this.loanAccountStatusListField;
            }
            set {
                this.loanAccountStatusListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/DepositFilter.xsd")]
    public partial class DepositFilter : AccountFilter {
        
        private DepositAccountStatus[] depositAccountStatusListField;
        
        private System.DateTime maturityStartDateField;
        
        private bool maturityStartDateFieldSpecified;
        
        private System.DateTime maturityEndDateField;
        
        private bool maturityEndDateFieldSpecified;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("depositAccountStatus", Namespace="http://cufxstandards.com/v3/Deposit.xsd", IsNullable=false)]
        public DepositAccountStatus[] depositAccountStatusList {
            get {
                return this.depositAccountStatusListField;
            }
            set {
                this.depositAccountStatusListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime maturityStartDate {
            get {
                return this.maturityStartDateField;
            }
            set {
                this.maturityStartDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool maturityStartDateSpecified {
            get {
                return this.maturityStartDateFieldSpecified;
            }
            set {
                this.maturityStartDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime maturityEndDate {
            get {
                return this.maturityEndDateField;
            }
            set {
                this.maturityEndDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool maturityEndDateSpecified {
            get {
                return this.maturityEndDateFieldSpecified;
            }
            set {
                this.maturityEndDateFieldSpecified = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/AccountMessage.xsd")]
    public partial class AccountMessage {
        
        private MessageContext messageContextField;
        
        private AccountFilter accountFilterField;
        
        private Account[] accountListField;
        
        /// <remarks/>
        public MessageContext messageContext {
            get {
                return this.messageContextField;
            }
            set {
                this.messageContextField = value;
            }
        }
        
        /// <remarks/>
        public AccountFilter accountFilter {
            get {
                return this.accountFilterField;
            }
            set {
                this.accountFilterField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("account", Namespace="http://cufxstandards.com/v3/Account.xsd", IsNullable=false)]
        public Account[] accountList {
            get {
                return this.accountListField;
            }
            set {
                this.accountListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/AnswerList.xsd")]
    public partial class AnswerList {
        
        private Answer[] answerField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("answer")]
        public Answer[] answer {
            get {
                return this.answerField;
            }
            set {
                this.answerField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/AnswerList.xsd")]
    public partial class Answer {
        
        private Party partyField;
        
        private string questionIdField;
        
        private Choice[] choiceListField;
        
        private Choice answerChoiceField;
        
        /// <remarks/>
        public Party party {
            get {
                return this.partyField;
            }
            set {
                this.partyField = value;
            }
        }
        
        /// <remarks/>
        public string questionId {
            get {
                return this.questionIdField;
            }
            set {
                this.questionIdField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("choice", Namespace="http://cufxstandards.com/v3/QuestionList.xsd", IsNullable=false)]
        public Choice[] choiceList {
            get {
                return this.choiceListField;
            }
            set {
                this.choiceListField = value;
            }
        }
        
        /// <remarks/>
        public Choice answerChoice {
            get {
                return this.answerChoiceField;
            }
            set {
                this.answerChoiceField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Party.xsd")]
    public partial class Party {
        
        private string idField;
        
        private Irs irsField;
        
        private PartyType typeField;
        
        private bool typeFieldSpecified;
        
        private Characteristics characteristicsField;
        
        private IdentificationDocument[] identificationDocumentListField;
        
        private EligibilityRequirementMet[] eligibilityRequirementMetListField;
        
        private string[] contactIdListField;
        
        private Contact[] contactListField;
        
        private string[] fiUserIdListField;
        
        private string householdIdField;
        
        private Note[] noteListField;
        
        private ValuePair[] customDataField;
        
        /// <remarks/>
        public string id {
            get {
                return this.idField;
            }
            set {
                this.idField = value;
            }
        }
        
        /// <remarks/>
        public Irs irs {
            get {
                return this.irsField;
            }
            set {
                this.irsField = value;
            }
        }
        
        /// <remarks/>
        public PartyType type {
            get {
                return this.typeField;
            }
            set {
                this.typeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool typeSpecified {
            get {
                return this.typeFieldSpecified;
            }
            set {
                this.typeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public Characteristics characteristics {
            get {
                return this.characteristicsField;
            }
            set {
                this.characteristicsField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("identificationDocument")]
        public IdentificationDocument[] identificationDocumentList {
            get {
                return this.identificationDocumentListField;
            }
            set {
                this.identificationDocumentListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("eligibilityRequirementMet")]
        public EligibilityRequirementMet[] eligibilityRequirementMetList {
            get {
                return this.eligibilityRequirementMetListField;
            }
            set {
                this.eligibilityRequirementMetListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("contactId", Namespace="http://cufxstandards.com/v3/Contact.xsd", IsNullable=false)]
        public string[] contactIdList {
            get {
                return this.contactIdListField;
            }
            set {
                this.contactIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("contact", Namespace="http://cufxstandards.com/v3/Contact.xsd", IsNullable=false)]
        public Contact[] contactList {
            get {
                return this.contactListField;
            }
            set {
                this.contactListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("fiUserId", Namespace="http://cufxstandards.com/v3/CredentialGroup.xsd", IsNullable=false)]
        public string[] fiUserIdList {
            get {
                return this.fiUserIdListField;
            }
            set {
                this.fiUserIdListField = value;
            }
        }
        
        /// <remarks/>
        public string householdId {
            get {
                return this.householdIdField;
            }
            set {
                this.householdIdField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("note", Namespace="http://cufxstandards.com/v3/Common.xsd", IsNullable=false)]
        public Note[] noteList {
            get {
                return this.noteListField;
            }
            set {
                this.noteListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("valuePair", Namespace="http://cufxstandards.com/v3/Common.xsd", IsNullable=false)]
        public ValuePair[] customData {
            get {
                return this.customDataField;
            }
            set {
                this.customDataField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Party.xsd")]
    public partial class Irs {
        
        private string taxIdField;
        
        private bool taxIdEncryptedField;
        
        private bool taxIdEncryptedFieldSpecified;
        
        private TaxIdType taxIdTypeField;
        
        private bool taxIdTypeFieldSpecified;
        
        private bool reportingFlagField;
        
        private bool verifiedTaxIdFlagField;
        
        private bool verifiedTaxIdFlagFieldSpecified;
        
        private string taxIdWarningCountField;
        
        private bool backupWithholdingFlagField;
        
        private bool backupWithholdingFlagFieldSpecified;
        
        private string backupWithholdingReasonField;
        
        private IrsBackupWithholdingExemptionReason backupWithholdingExemptionReasonField;
        
        private bool backupWithholdingExemptionReasonFieldSpecified;
        
        private System.DateTime backupWithholdingEffectiveDateField;
        
        private bool backupWithholdingEffectiveDateFieldSpecified;
        
        public Irs() {
            this.reportingFlagField = true;
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="token")]
        public string taxId {
            get {
                return this.taxIdField;
            }
            set {
                this.taxIdField = value;
            }
        }
        
        /// <remarks/>
        public bool taxIdEncrypted {
            get {
                return this.taxIdEncryptedField;
            }
            set {
                this.taxIdEncryptedField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool taxIdEncryptedSpecified {
            get {
                return this.taxIdEncryptedFieldSpecified;
            }
            set {
                this.taxIdEncryptedFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public TaxIdType taxIdType {
            get {
                return this.taxIdTypeField;
            }
            set {
                this.taxIdTypeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool taxIdTypeSpecified {
            get {
                return this.taxIdTypeFieldSpecified;
            }
            set {
                this.taxIdTypeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.ComponentModel.DefaultValueAttribute(true)]
        public bool reportingFlag {
            get {
                return this.reportingFlagField;
            }
            set {
                this.reportingFlagField = value;
            }
        }
        
        /// <remarks/>
        public bool verifiedTaxIdFlag {
            get {
                return this.verifiedTaxIdFlagField;
            }
            set {
                this.verifiedTaxIdFlagField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool verifiedTaxIdFlagSpecified {
            get {
                return this.verifiedTaxIdFlagFieldSpecified;
            }
            set {
                this.verifiedTaxIdFlagFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="nonNegativeInteger")]
        public string taxIdWarningCount {
            get {
                return this.taxIdWarningCountField;
            }
            set {
                this.taxIdWarningCountField = value;
            }
        }
        
        /// <remarks/>
        public bool backupWithholdingFlag {
            get {
                return this.backupWithholdingFlagField;
            }
            set {
                this.backupWithholdingFlagField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool backupWithholdingFlagSpecified {
            get {
                return this.backupWithholdingFlagFieldSpecified;
            }
            set {
                this.backupWithholdingFlagFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string backupWithholdingReason {
            get {
                return this.backupWithholdingReasonField;
            }
            set {
                this.backupWithholdingReasonField = value;
            }
        }
        
        /// <remarks/>
        public IrsBackupWithholdingExemptionReason backupWithholdingExemptionReason {
            get {
                return this.backupWithholdingExemptionReasonField;
            }
            set {
                this.backupWithholdingExemptionReasonField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool backupWithholdingExemptionReasonSpecified {
            get {
                return this.backupWithholdingExemptionReasonFieldSpecified;
            }
            set {
                this.backupWithholdingExemptionReasonFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime backupWithholdingEffectiveDate {
            get {
                return this.backupWithholdingEffectiveDateField;
            }
            set {
                this.backupWithholdingEffectiveDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool backupWithholdingEffectiveDateSpecified {
            get {
                return this.backupWithholdingEffectiveDateFieldSpecified;
            }
            set {
                this.backupWithholdingEffectiveDateFieldSpecified = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Party.xsd")]
    public enum TaxIdType {
        
        /// <remarks/>
        SocialSecurityNumber,
        
        /// <remarks/>
        EmployerIdentificationNumber,
        
        /// <remarks/>
        IndividualTaxpayerIdentificationNumber,
        
        /// <remarks/>
        TaxpayerIdentificationNumberForPendingUSAdoptions,
        
        /// <remarks/>
        PreparerTaxpayerIdentificationNumber,
        
        /// <remarks/>
        ForeignNational,
        
        /// <remarks/>
        ForeignNumberNoTIN,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(AnonymousType=true, Namespace="http://cufxstandards.com/v3/Party.xsd")]
    public enum IrsBackupWithholdingExemptionReason {
        
        /// <remarks/>
        ExceptFromWithholding,
        
        /// <remarks/>
        NotNotifiedByIrs,
        
        /// <remarks/>
        IrsNotifiedNoLongSubject,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Party.xsd")]
    public enum PartyType {
        
        /// <remarks/>
        Individual,
        
        /// <remarks/>
        Organization,
        
        /// <remarks/>
        Trust,
        
        /// <remarks/>
        Estate,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Party.xsd")]
    public partial class Characteristics {
        
        private object itemField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("estate", typeof(Estate))]
        [System.Xml.Serialization.XmlElementAttribute("individual", typeof(Individual))]
        [System.Xml.Serialization.XmlElementAttribute("organization", typeof(Organization))]
        [System.Xml.Serialization.XmlElementAttribute("trust", typeof(Trust))]
        public object Item {
            get {
                return this.itemField;
            }
            set {
                this.itemField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Party.xsd")]
    public partial class Estate {
        
        private string estateNameField;
        
        /// <remarks/>
        public string estateName {
            get {
                return this.estateNameField;
            }
            set {
                this.estateNameField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Party.xsd")]
    public partial class Individual {
        
        private string firstNameField;
        
        private string middleNameField;
        
        private string lastNameField;
        
        private string prefixField;
        
        private string suffixField;
        
        private string formattedNameField;
        
        private string mothersMaidenNameField;
        
        private string nicknameField;
        
        private System.DateTime birthdateField;
        
        private bool birthdateFieldSpecified;
        
        private System.DateTime deathDateField;
        
        private bool deathDateFieldSpecified;
        
        private IndividualGender genderField;
        
        private bool genderFieldSpecified;
        
        private string cityOfBirthField;
        
        private Citizenship[] citizenshipListField;
        
        private EmploymentStatusType employmentStatusField;
        
        private bool employmentStatusFieldSpecified;
        
        private Employment[] employmentListField;
        
        private AdditionalIncome[] additionalIncomeListField;
        
        private Liability[] liabilityListField;
        
        private Residence residenceField;
        
        private string ethnicityField;
        
        private string raceField;
        
        /// <remarks/>
        public string firstName {
            get {
                return this.firstNameField;
            }
            set {
                this.firstNameField = value;
            }
        }
        
        /// <remarks/>
        public string middleName {
            get {
                return this.middleNameField;
            }
            set {
                this.middleNameField = value;
            }
        }
        
        /// <remarks/>
        public string lastName {
            get {
                return this.lastNameField;
            }
            set {
                this.lastNameField = value;
            }
        }
        
        /// <remarks/>
        public string prefix {
            get {
                return this.prefixField;
            }
            set {
                this.prefixField = value;
            }
        }
        
        /// <remarks/>
        public string suffix {
            get {
                return this.suffixField;
            }
            set {
                this.suffixField = value;
            }
        }
        
        /// <remarks/>
        public string formattedName {
            get {
                return this.formattedNameField;
            }
            set {
                this.formattedNameField = value;
            }
        }
        
        /// <remarks/>
        public string mothersMaidenName {
            get {
                return this.mothersMaidenNameField;
            }
            set {
                this.mothersMaidenNameField = value;
            }
        }
        
        /// <remarks/>
        public string nickname {
            get {
                return this.nicknameField;
            }
            set {
                this.nicknameField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime birthdate {
            get {
                return this.birthdateField;
            }
            set {
                this.birthdateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool birthdateSpecified {
            get {
                return this.birthdateFieldSpecified;
            }
            set {
                this.birthdateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime deathDate {
            get {
                return this.deathDateField;
            }
            set {
                this.deathDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool deathDateSpecified {
            get {
                return this.deathDateFieldSpecified;
            }
            set {
                this.deathDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public IndividualGender gender {
            get {
                return this.genderField;
            }
            set {
                this.genderField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool genderSpecified {
            get {
                return this.genderFieldSpecified;
            }
            set {
                this.genderFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string cityOfBirth {
            get {
                return this.cityOfBirthField;
            }
            set {
                this.cityOfBirthField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("citizenship")]
        public Citizenship[] citizenshipList {
            get {
                return this.citizenshipListField;
            }
            set {
                this.citizenshipListField = value;
            }
        }
        
        /// <remarks/>
        public EmploymentStatusType employmentStatus {
            get {
                return this.employmentStatusField;
            }
            set {
                this.employmentStatusField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool employmentStatusSpecified {
            get {
                return this.employmentStatusFieldSpecified;
            }
            set {
                this.employmentStatusFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("employment", IsNullable=false)]
        public Employment[] employmentList {
            get {
                return this.employmentListField;
            }
            set {
                this.employmentListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("additionalIncome", IsNullable=false)]
        public AdditionalIncome[] additionalIncomeList {
            get {
                return this.additionalIncomeListField;
            }
            set {
                this.additionalIncomeListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("liability", IsNullable=false)]
        public Liability[] liabilityList {
            get {
                return this.liabilityListField;
            }
            set {
                this.liabilityListField = value;
            }
        }
        
        /// <remarks/>
        public Residence residence {
            get {
                return this.residenceField;
            }
            set {
                this.residenceField = value;
            }
        }
        
        /// <remarks/>
        public string ethnicity {
            get {
                return this.ethnicityField;
            }
            set {
                this.ethnicityField = value;
            }
        }
        
        /// <remarks/>
        public string race {
            get {
                return this.raceField;
            }
            set {
                this.raceField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(AnonymousType=true, Namespace="http://cufxstandards.com/v3/Party.xsd")]
    public enum IndividualGender {
        
        /// <remarks/>
        Male,
        
        /// <remarks/>
        Female,
        
        /// <remarks/>
        Other,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Party.xsd")]
    public partial class Citizenship {
        
        private System.Nullable<ISOCountryCodeType> citizenshipField;
        
        private bool citizenshipFieldSpecified;
        
        private bool wasCitizenshipCertifiedField;
        
        private bool wasCitizenshipCertifiedFieldSpecified;
        
        private ValuePair[] customDataField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(IsNullable=true)]
        public System.Nullable<ISOCountryCodeType> citizenship {
            get {
                return this.citizenshipField;
            }
            set {
                this.citizenshipField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool citizenshipSpecified {
            get {
                return this.citizenshipFieldSpecified;
            }
            set {
                this.citizenshipFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public bool wasCitizenshipCertified {
            get {
                return this.wasCitizenshipCertifiedField;
            }
            set {
                this.wasCitizenshipCertifiedField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool wasCitizenshipCertifiedSpecified {
            get {
                return this.wasCitizenshipCertifiedFieldSpecified;
            }
            set {
                this.wasCitizenshipCertifiedFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("valuePair", Namespace="http://cufxstandards.com/v3/Common.xsd", IsNullable=false)]
        public ValuePair[] customData {
            get {
                return this.customDataField;
            }
            set {
                this.customDataField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Party.xsd")]
    public enum EmploymentStatusType {
        
        /// <remarks/>
        Contract,
        
        /// <remarks/>
        Employed,
        
        /// <remarks/>
        Homemaker,
        
        /// <remarks/>
        Other,
        
        /// <remarks/>
        Retired,
        
        /// <remarks/>
        SelfEmployed,
        
        /// <remarks/>
        Student,
        
        /// <remarks/>
        Temporary,
        
        /// <remarks/>
        Unemployed,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Party.xsd")]
    public partial class Employment {
        
        private string employmentIdField;
        
        private string employerPartyIdField;
        
        private string employerNameField;
        
        private System.DateTime employmentStartDateField;
        
        private bool employmentStartDateFieldSpecified;
        
        private System.DateTime employmentEndDateField;
        
        private bool employmentEndDateFieldSpecified;
        
        private string timeAtEmployerField;
        
        private string employeeOccupationField;
        
        private Address[] employerAddressField;
        
        private Phone[] employerPhoneField;
        
        private EmployerStatusType employerStatusField;
        
        private bool employerStatusFieldSpecified;
        
        private EmploymentType typeField;
        
        private bool typeFieldSpecified;
        
        private IncomeDetail employmentIncomeField;
        
        private ValuePair[] customDataField;
        
        /// <remarks/>
        public string employmentId {
            get {
                return this.employmentIdField;
            }
            set {
                this.employmentIdField = value;
            }
        }
        
        /// <remarks/>
        public string employerPartyId {
            get {
                return this.employerPartyIdField;
            }
            set {
                this.employerPartyIdField = value;
            }
        }
        
        /// <remarks/>
        public string employerName {
            get {
                return this.employerNameField;
            }
            set {
                this.employerNameField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime employmentStartDate {
            get {
                return this.employmentStartDateField;
            }
            set {
                this.employmentStartDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool employmentStartDateSpecified {
            get {
                return this.employmentStartDateFieldSpecified;
            }
            set {
                this.employmentStartDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime employmentEndDate {
            get {
                return this.employmentEndDateField;
            }
            set {
                this.employmentEndDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool employmentEndDateSpecified {
            get {
                return this.employmentEndDateFieldSpecified;
            }
            set {
                this.employmentEndDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="duration")]
        public string timeAtEmployer {
            get {
                return this.timeAtEmployerField;
            }
            set {
                this.timeAtEmployerField = value;
            }
        }
        
        /// <remarks/>
        public string employeeOccupation {
            get {
                return this.employeeOccupationField;
            }
            set {
                this.employeeOccupationField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("employerAddress")]
        public Address[] employerAddress {
            get {
                return this.employerAddressField;
            }
            set {
                this.employerAddressField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("employerPhone")]
        public Phone[] employerPhone {
            get {
                return this.employerPhoneField;
            }
            set {
                this.employerPhoneField = value;
            }
        }
        
        /// <remarks/>
        public EmployerStatusType employerStatus {
            get {
                return this.employerStatusField;
            }
            set {
                this.employerStatusField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool employerStatusSpecified {
            get {
                return this.employerStatusFieldSpecified;
            }
            set {
                this.employerStatusFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public EmploymentType type {
            get {
                return this.typeField;
            }
            set {
                this.typeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool typeSpecified {
            get {
                return this.typeFieldSpecified;
            }
            set {
                this.typeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public IncomeDetail employmentIncome {
            get {
                return this.employmentIncomeField;
            }
            set {
                this.employmentIncomeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("valuePair", Namespace="http://cufxstandards.com/v3/Common.xsd", IsNullable=false)]
        public ValuePair[] customData {
            get {
                return this.customDataField;
            }
            set {
                this.customDataField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Contact.xsd")]
    public partial class Phone {
        
        private PhoneType typeField;
        
        private bool typeFieldSpecified;
        
        private string numberField;
        
        private string extensionField;
        
        private PlanFormat planFormatField;
        
        private bool planFormatFieldSpecified;
        
        private string descriptionField;
        
        private bool smsRegisteredField;
        
        private bool smsRegisteredFieldSpecified;
        
        /// <remarks/>
        public PhoneType type {
            get {
                return this.typeField;
            }
            set {
                this.typeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool typeSpecified {
            get {
                return this.typeFieldSpecified;
            }
            set {
                this.typeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string number {
            get {
                return this.numberField;
            }
            set {
                this.numberField = value;
            }
        }
        
        /// <remarks/>
        public string extension {
            get {
                return this.extensionField;
            }
            set {
                this.extensionField = value;
            }
        }
        
        /// <remarks/>
        public PlanFormat planFormat {
            get {
                return this.planFormatField;
            }
            set {
                this.planFormatField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool planFormatSpecified {
            get {
                return this.planFormatFieldSpecified;
            }
            set {
                this.planFormatFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string description {
            get {
                return this.descriptionField;
            }
            set {
                this.descriptionField = value;
            }
        }
        
        /// <remarks/>
        public bool smsRegistered {
            get {
                return this.smsRegisteredField;
            }
            set {
                this.smsRegisteredField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool smsRegisteredSpecified {
            get {
                return this.smsRegisteredFieldSpecified;
            }
            set {
                this.smsRegisteredFieldSpecified = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(AnonymousType=true, Namespace="http://cufxstandards.com/v3/Contact.xsd")]
    public enum PhoneType {
        
        /// <remarks/>
        Home,
        
        /// <remarks/>
        Work,
        
        /// <remarks/>
        Mobile,
        
        /// <remarks/>
        Pager,
        
        /// <remarks/>
        Fax,
        
        /// <remarks/>
        Other,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Contact.xsd")]
    public enum PlanFormat {
        
        /// <remarks/>
        Nanp,
        
        /// <remarks/>
        NanpLessCountryCode,
        
        /// <remarks/>
        OtherInternational,
        
        /// <remarks/>
        NationalNumber,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Party.xsd")]
    public enum EmployerStatusType {
        
        /// <remarks/>
        Current,
        
        /// <remarks/>
        Previous,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Party.xsd")]
    public enum EmploymentType {
        
        /// <remarks/>
        FullTime,
        
        /// <remarks/>
        PartTime,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Party.xsd")]
    public partial class IncomeDetail {
        
        private Income grossIncomeDataField;
        
        private Income netIncomeDataField;
        
        /// <remarks/>
        public Income grossIncomeData {
            get {
                return this.grossIncomeDataField;
            }
            set {
                this.grossIncomeDataField = value;
            }
        }
        
        /// <remarks/>
        public Income netIncomeData {
            get {
                return this.netIncomeDataField;
            }
            set {
                this.netIncomeDataField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Party.xsd")]
    public partial class Income {
        
        private Money amountField;
        
        private FrequencyType frequencyField;
        
        private bool frequencyFieldSpecified;
        
        private string otherFrequencyTypeField;
        
        private bool excludeIncomeFromCalculationsField;
        
        private ValuePair[] customDataField;
        
        public Income() {
            this.excludeIncomeFromCalculationsField = false;
        }
        
        /// <remarks/>
        public Money amount {
            get {
                return this.amountField;
            }
            set {
                this.amountField = value;
            }
        }
        
        /// <remarks/>
        public FrequencyType frequency {
            get {
                return this.frequencyField;
            }
            set {
                this.frequencyField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool frequencySpecified {
            get {
                return this.frequencyFieldSpecified;
            }
            set {
                this.frequencyFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string otherFrequencyType {
            get {
                return this.otherFrequencyTypeField;
            }
            set {
                this.otherFrequencyTypeField = value;
            }
        }
        
        /// <remarks/>
        [System.ComponentModel.DefaultValueAttribute(false)]
        public bool excludeIncomeFromCalculations {
            get {
                return this.excludeIncomeFromCalculationsField;
            }
            set {
                this.excludeIncomeFromCalculationsField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("valuePair", Namespace="http://cufxstandards.com/v3/Common.xsd", IsNullable=false)]
        public ValuePair[] customData {
            get {
                return this.customDataField;
            }
            set {
                this.customDataField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Party.xsd")]
    public enum FrequencyType {
        
        /// <remarks/>
        Biweekly,
        
        /// <remarks/>
        Monthly,
        
        /// <remarks/>
        SemiMonthly,
        
        /// <remarks/>
        Other,
        
        /// <remarks/>
        Weekly,
        
        /// <remarks/>
        Yearly,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Party.xsd")]
    public partial class AdditionalIncome : Income {
        
        private string sourceField;
        
        /// <remarks/>
        public string source {
            get {
                return this.sourceField;
            }
            set {
                this.sourceField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Party.xsd")]
    public partial class Liability {
        
        private string descriptionField;
        
        private Money paymentField;
        
        private FrequencyType paymentFrequencyField;
        
        private bool paymentFrequencyFieldSpecified;
        
        private Money balanceField;
        
        private bool excludeLiabilityFromCalculationsField;
        
        private bool excludeLiabilityFromCalculationsFieldSpecified;
        
        private ValuePair[] customDataField;
        
        /// <remarks/>
        public string description {
            get {
                return this.descriptionField;
            }
            set {
                this.descriptionField = value;
            }
        }
        
        /// <remarks/>
        public Money payment {
            get {
                return this.paymentField;
            }
            set {
                this.paymentField = value;
            }
        }
        
        /// <remarks/>
        public FrequencyType paymentFrequency {
            get {
                return this.paymentFrequencyField;
            }
            set {
                this.paymentFrequencyField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool paymentFrequencySpecified {
            get {
                return this.paymentFrequencyFieldSpecified;
            }
            set {
                this.paymentFrequencyFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public Money balance {
            get {
                return this.balanceField;
            }
            set {
                this.balanceField = value;
            }
        }
        
        /// <remarks/>
        public bool excludeLiabilityFromCalculations {
            get {
                return this.excludeLiabilityFromCalculationsField;
            }
            set {
                this.excludeLiabilityFromCalculationsField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool excludeLiabilityFromCalculationsSpecified {
            get {
                return this.excludeLiabilityFromCalculationsFieldSpecified;
            }
            set {
                this.excludeLiabilityFromCalculationsFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("valuePair", Namespace="http://cufxstandards.com/v3/Common.xsd", IsNullable=false)]
        public ValuePair[] customData {
            get {
                return this.customDataField;
            }
            set {
                this.customDataField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Party.xsd")]
    public partial class Residence {
        
        private HousingStatusType currentHousingStatusField;
        
        private bool currentHousingStatusFieldSpecified;
        
        private HousingOwnershipType currentHousingOwnershipTypeField;
        
        private bool currentHousingOwnershipTypeFieldSpecified;
        
        private HousingDebtType currentHousingDebtTypeField;
        
        private bool currentHousingDebtTypeFieldSpecified;
        
        private HousingType currentHousingTypeField;
        
        private bool currentHousingTypeFieldSpecified;
        
        /// <remarks/>
        public HousingStatusType currentHousingStatus {
            get {
                return this.currentHousingStatusField;
            }
            set {
                this.currentHousingStatusField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool currentHousingStatusSpecified {
            get {
                return this.currentHousingStatusFieldSpecified;
            }
            set {
                this.currentHousingStatusFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public HousingOwnershipType currentHousingOwnershipType {
            get {
                return this.currentHousingOwnershipTypeField;
            }
            set {
                this.currentHousingOwnershipTypeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool currentHousingOwnershipTypeSpecified {
            get {
                return this.currentHousingOwnershipTypeFieldSpecified;
            }
            set {
                this.currentHousingOwnershipTypeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public HousingDebtType currentHousingDebtType {
            get {
                return this.currentHousingDebtTypeField;
            }
            set {
                this.currentHousingDebtTypeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool currentHousingDebtTypeSpecified {
            get {
                return this.currentHousingDebtTypeFieldSpecified;
            }
            set {
                this.currentHousingDebtTypeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public HousingType currentHousingType {
            get {
                return this.currentHousingTypeField;
            }
            set {
                this.currentHousingTypeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool currentHousingTypeSpecified {
            get {
                return this.currentHousingTypeFieldSpecified;
            }
            set {
                this.currentHousingTypeFieldSpecified = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Party.xsd")]
    public enum HousingStatusType {
        
        /// <remarks/>
        Rent,
        
        /// <remarks/>
        Own,
        
        /// <remarks/>
        Other,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Party.xsd")]
    public enum HousingOwnershipType {
        
        /// <remarks/>
        Self,
        
        /// <remarks/>
        Government,
        
        /// <remarks/>
        Military,
        
        /// <remarks/>
        Other,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Party.xsd")]
    public enum HousingDebtType {
        
        /// <remarks/>
        OwnWithMortgage,
        
        /// <remarks/>
        OwnWithoutMortgage,
        
        /// <remarks/>
        Rent,
        
        /// <remarks/>
        NoHousingExpense,
        
        /// <remarks/>
        Other,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Party.xsd")]
    public enum HousingType {
        
        /// <remarks/>
        PrimaryResidence,
        
        /// <remarks/>
        NonPrimaryResidence,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Party.xsd")]
    public partial class Organization {
        
        private OrganizationType organizationTypeField;
        
        private bool organizationTypeFieldSpecified;
        
        private string organizationNameField;
        
        private DoingBusinessAs[] doingBusinessAsListField;
        
        /// <remarks/>
        public OrganizationType organizationType {
            get {
                return this.organizationTypeField;
            }
            set {
                this.organizationTypeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool organizationTypeSpecified {
            get {
                return this.organizationTypeFieldSpecified;
            }
            set {
                this.organizationTypeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string organizationName {
            get {
                return this.organizationNameField;
            }
            set {
                this.organizationNameField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("doingBusinessAs")]
        public DoingBusinessAs[] doingBusinessAsList {
            get {
                return this.doingBusinessAsListField;
            }
            set {
                this.doingBusinessAsListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Party.xsd")]
    public enum OrganizationType {
        
        /// <remarks/>
        SoleProprietorship,
        
        /// <remarks/>
        Llc,
        
        /// <remarks/>
        Partnership,
        
        /// <remarks/>
        Corporation,
        
        /// <remarks/>
        NotForProfit,
        
        /// <remarks/>
        Club,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Party.xsd")]
    public partial class DoingBusinessAs {
        
        private string doingBusinessAsIdField;
        
        private string doingBusinessAsNameField;
        
        private ValuePair[] customDataField;
        
        /// <remarks/>
        public string doingBusinessAsId {
            get {
                return this.doingBusinessAsIdField;
            }
            set {
                this.doingBusinessAsIdField = value;
            }
        }
        
        /// <remarks/>
        public string doingBusinessAsName {
            get {
                return this.doingBusinessAsNameField;
            }
            set {
                this.doingBusinessAsNameField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("valuePair", Namespace="http://cufxstandards.com/v3/Common.xsd", IsNullable=false)]
        public ValuePair[] customData {
            get {
                return this.customDataField;
            }
            set {
                this.customDataField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Party.xsd")]
    public partial class Trust {
        
        private string trustNameField;
        
        /// <remarks/>
        public string trustName {
            get {
                return this.trustNameField;
            }
            set {
                this.trustNameField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Party.xsd")]
    public partial class IdentificationDocument {
        
        private string idDocumentIdentiferField;
        
        private IdDocumentType idDocumentTypeField;
        
        private string idIssuedByField;
        
        private System.DateTime idIssueDateField;
        
        private bool idIssueDateFieldSpecified;
        
        private System.DateTime idExpirationDateField;
        
        private bool idExpirationDateFieldSpecified;
        
        private string idDisplayOrderField;
        
        private System.DateTime idVerifyDateTimeField;
        
        private bool idVerifyDateTimeFieldSpecified;
        
        private string documentIdField;
        
        /// <remarks/>
        public string idDocumentIdentifer {
            get {
                return this.idDocumentIdentiferField;
            }
            set {
                this.idDocumentIdentiferField = value;
            }
        }
        
        /// <remarks/>
        public IdDocumentType idDocumentType {
            get {
                return this.idDocumentTypeField;
            }
            set {
                this.idDocumentTypeField = value;
            }
        }
        
        /// <remarks/>
        public string idIssuedBy {
            get {
                return this.idIssuedByField;
            }
            set {
                this.idIssuedByField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime idIssueDate {
            get {
                return this.idIssueDateField;
            }
            set {
                this.idIssueDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool idIssueDateSpecified {
            get {
                return this.idIssueDateFieldSpecified;
            }
            set {
                this.idIssueDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime idExpirationDate {
            get {
                return this.idExpirationDateField;
            }
            set {
                this.idExpirationDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool idExpirationDateSpecified {
            get {
                return this.idExpirationDateFieldSpecified;
            }
            set {
                this.idExpirationDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="nonNegativeInteger")]
        public string idDisplayOrder {
            get {
                return this.idDisplayOrderField;
            }
            set {
                this.idDisplayOrderField = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime idVerifyDateTime {
            get {
                return this.idVerifyDateTimeField;
            }
            set {
                this.idVerifyDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool idVerifyDateTimeSpecified {
            get {
                return this.idVerifyDateTimeFieldSpecified;
            }
            set {
                this.idVerifyDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string documentId {
            get {
                return this.documentIdField;
            }
            set {
                this.documentIdField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Party.xsd")]
    public partial class IdDocumentType {
        
        private object itemField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("estateDocument", typeof(IdDocumentTypeEstateDocument))]
        [System.Xml.Serialization.XmlElementAttribute("individualDocument", typeof(IdDocumentTypeIndividualDocument))]
        [System.Xml.Serialization.XmlElementAttribute("organizationDocument", typeof(IdDocumentTypeOrganizationDocument))]
        [System.Xml.Serialization.XmlElementAttribute("trustDocument", typeof(IdDocumentTypeTrustDocument))]
        public object Item {
            get {
                return this.itemField;
            }
            set {
                this.itemField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(AnonymousType=true, Namespace="http://cufxstandards.com/v3/Party.xsd")]
    public enum IdDocumentTypeEstateDocument {
        
        /// <remarks/>
        Will,
        
        /// <remarks/>
        EstateInstrument,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(AnonymousType=true, Namespace="http://cufxstandards.com/v3/Party.xsd")]
    public enum IdDocumentTypeIndividualDocument {
        
        /// <remarks/>
        DriversLicense,
        
        /// <remarks/>
        USPassport,
        
        /// <remarks/>
        MilitaryId,
        
        /// <remarks/>
        StateIssuedId,
        
        /// <remarks/>
        BirthCertficate,
        
        /// <remarks/>
        ForeignPassport,
        
        /// <remarks/>
        KnownExistingParty,
        
        /// <remarks/>
        ForeignGovernmentId,
        
        /// <remarks/>
        ResidentAlienCard,
        
        /// <remarks/>
        NonResidentAlienCard,
        
        /// <remarks/>
        DisabledElderlyWithNoId,
        
        /// <remarks/>
        ForeignEntityWithNoId,
        
        /// <remarks/>
        LawEnforcementId,
        
        /// <remarks/>
        AmishPartyWithNoId,
        
        /// <remarks/>
        ForeignDriversLicense,
        
        /// <remarks/>
        InsuranceCard,
        
        /// <remarks/>
        OrganizationalMembershipCard,
        
        /// <remarks/>
        PropertyTaxBill,
        
        /// <remarks/>
        SocialSecurityCard,
        
        /// <remarks/>
        StudentId,
        
        /// <remarks/>
        UtilityBill,
        
        /// <remarks/>
        Visa,
        
        /// <remarks/>
        DepartmentHomelandSecurityEmploymentAuthorization,
        
        /// <remarks/>
        VoterRegistrationCard,
        
        /// <remarks/>
        Photo,
        
        /// <remarks/>
        Other,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(AnonymousType=true, Namespace="http://cufxstandards.com/v3/Party.xsd")]
    public enum IdDocumentTypeOrganizationDocument {
        
        /// <remarks/>
        StateCorporateId,
        
        /// <remarks/>
        StateDba,
        
        /// <remarks/>
        ArticlesOfIncorporation,
        
        /// <remarks/>
        BusinessLicense,
        
        /// <remarks/>
        CorporateResolution,
        
        /// <remarks/>
        SecretaryOfStateFilingReceipt,
        
        /// <remarks/>
        AssociationMinutes,
        
        /// <remarks/>
        PartnershipAgreement,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(AnonymousType=true, Namespace="http://cufxstandards.com/v3/Party.xsd")]
    public enum IdDocumentTypeTrustDocument {
        
        /// <remarks/>
        TrustDocument,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Party.xsd")]
    public partial class EligibilityRequirementMet {
        
        private string requirementIdField;
        
        private string referenceDescriptionField;
        
        private ValuePair[] customDataField;
        
        /// <remarks/>
        public string requirementId {
            get {
                return this.requirementIdField;
            }
            set {
                this.requirementIdField = value;
            }
        }
        
        /// <remarks/>
        public string referenceDescription {
            get {
                return this.referenceDescriptionField;
            }
            set {
                this.referenceDescriptionField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("valuePair", Namespace="http://cufxstandards.com/v3/Common.xsd", IsNullable=false)]
        public ValuePair[] customData {
            get {
                return this.customDataField;
            }
            set {
                this.customDataField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Contact.xsd")]
    public partial class Contact {
        
        private string contactIdField;
        
        private ContactType contactTypeField;
        
        private ContactAddress addressField;
        
        private Phone phoneField;
        
        private Email emailField;
        
        private InstantMessage instantMessageField;
        
        private SocialContactPoint socialField;
        
        private Website websiteField;
        
        private TimeOfDayType timeOfDayField;
        
        private bool timeOfDayFieldSpecified;
        
        private DemonstratedAccess demonstratedAccessField;
        
        private bool badContactPointField;
        
        private bool badContactPointFieldSpecified;
        
        private ValuePair[] customDataField;
        
        /// <remarks/>
        public string contactId {
            get {
                return this.contactIdField;
            }
            set {
                this.contactIdField = value;
            }
        }
        
        /// <remarks/>
        public ContactType contactType {
            get {
                return this.contactTypeField;
            }
            set {
                this.contactTypeField = value;
            }
        }
        
        /// <remarks/>
        public ContactAddress address {
            get {
                return this.addressField;
            }
            set {
                this.addressField = value;
            }
        }
        
        /// <remarks/>
        public Phone phone {
            get {
                return this.phoneField;
            }
            set {
                this.phoneField = value;
            }
        }
        
        /// <remarks/>
        public Email email {
            get {
                return this.emailField;
            }
            set {
                this.emailField = value;
            }
        }
        
        /// <remarks/>
        public InstantMessage instantMessage {
            get {
                return this.instantMessageField;
            }
            set {
                this.instantMessageField = value;
            }
        }
        
        /// <remarks/>
        public SocialContactPoint social {
            get {
                return this.socialField;
            }
            set {
                this.socialField = value;
            }
        }
        
        /// <remarks/>
        public Website website {
            get {
                return this.websiteField;
            }
            set {
                this.websiteField = value;
            }
        }
        
        /// <remarks/>
        public TimeOfDayType timeOfDay {
            get {
                return this.timeOfDayField;
            }
            set {
                this.timeOfDayField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool timeOfDaySpecified {
            get {
                return this.timeOfDayFieldSpecified;
            }
            set {
                this.timeOfDayFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public DemonstratedAccess demonstratedAccess {
            get {
                return this.demonstratedAccessField;
            }
            set {
                this.demonstratedAccessField = value;
            }
        }
        
        /// <remarks/>
        public bool badContactPoint {
            get {
                return this.badContactPointField;
            }
            set {
                this.badContactPointField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool badContactPointSpecified {
            get {
                return this.badContactPointFieldSpecified;
            }
            set {
                this.badContactPointFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("valuePair", Namespace="http://cufxstandards.com/v3/Common.xsd", IsNullable=false)]
        public ValuePair[] customData {
            get {
                return this.customDataField;
            }
            set {
                this.customDataField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Contact.xsd")]
    public enum ContactType {
        
        /// <remarks/>
        Address,
        
        /// <remarks/>
        Phone,
        
        /// <remarks/>
        Email,
        
        /// <remarks/>
        InstantMessaging,
        
        /// <remarks/>
        Social,
        
        /// <remarks/>
        Website,
        
        /// <remarks/>
        Other,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Contact.xsd")]
    public partial class Email {
        
        private EmailType typeField;
        
        private bool typeFieldSpecified;
        
        private string addressField;
        
        /// <remarks/>
        public EmailType type {
            get {
                return this.typeField;
            }
            set {
                this.typeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool typeSpecified {
            get {
                return this.typeFieldSpecified;
            }
            set {
                this.typeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string address {
            get {
                return this.addressField;
            }
            set {
                this.addressField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(AnonymousType=true, Namespace="http://cufxstandards.com/v3/Contact.xsd")]
    public enum EmailType {
        
        /// <remarks/>
        Home,
        
        /// <remarks/>
        Work,
        
        /// <remarks/>
        Other,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Contact.xsd")]
    public partial class InstantMessage {
        
        private string messagingServiceField;
        
        private string userField;
        
        /// <remarks/>
        public string messagingService {
            get {
                return this.messagingServiceField;
            }
            set {
                this.messagingServiceField = value;
            }
        }
        
        /// <remarks/>
        public string user {
            get {
                return this.userField;
            }
            set {
                this.userField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Contact.xsd")]
    public partial class SocialContactPoint {
        
        private string socialServiceField;
        
        private string userField;
        
        /// <remarks/>
        public string socialService {
            get {
                return this.socialServiceField;
            }
            set {
                this.socialServiceField = value;
            }
        }
        
        /// <remarks/>
        public string user {
            get {
                return this.userField;
            }
            set {
                this.userField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Contact.xsd")]
    public partial class Website {
        
        private string sitenameField;
        
        private string userField;
        
        /// <remarks/>
        public string sitename {
            get {
                return this.sitenameField;
            }
            set {
                this.sitenameField = value;
            }
        }
        
        /// <remarks/>
        public string user {
            get {
                return this.userField;
            }
            set {
                this.userField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Contact.xsd")]
    public enum TimeOfDayType {
        
        /// <remarks/>
        Afternoon,
        
        /// <remarks/>
        Evening,
        
        /// <remarks/>
        Morning,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Contact.xsd")]
    public partial class DemonstratedAccess {
        
        private System.DateTime dateTimeField;
        
        private bool dateTimeFieldSpecified;
        
        private string fullNameField;
        
        private string userNameField;
        
        /// <remarks/>
        public System.DateTime dateTime {
            get {
                return this.dateTimeField;
            }
            set {
                this.dateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool dateTimeSpecified {
            get {
                return this.dateTimeFieldSpecified;
            }
            set {
                this.dateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string fullName {
            get {
                return this.fullNameField;
            }
            set {
                this.fullNameField = value;
            }
        }
        
        /// <remarks/>
        public string userName {
            get {
                return this.userNameField;
            }
            set {
                this.userNameField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/QuestionList.xsd")]
    public partial class Choice {
        
        private string choiceIdField;
        
        private string choiceTextField;
        
        private ValuePair[] customDataField;
        
        /// <remarks/>
        public string choiceId {
            get {
                return this.choiceIdField;
            }
            set {
                this.choiceIdField = value;
            }
        }
        
        /// <remarks/>
        public string choiceText {
            get {
                return this.choiceTextField;
            }
            set {
                this.choiceTextField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("valuePair", Namespace="http://cufxstandards.com/v3/Common.xsd", IsNullable=false)]
        public ValuePair[] customData {
            get {
                return this.customDataField;
            }
            set {
                this.customDataField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Application.xsd")]
    public partial class ApplicationList {
        
        private Application[] applicationField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("application")]
        public Application[] application {
            get {
                return this.applicationField;
            }
            set {
                this.applicationField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Application.xsd")]
    public partial class Application {
        
        private string applicationIdField;
        
        private System.DateTime applicationDateField;
        
        private bool applicationDateFieldSpecified;
        
        private ApplicationStatus applicationStatusField;
        
        private bool applicationStatusFieldSpecified;
        
        private ProductAppliedFor[] productAppliedForListField;
        
        private Applicant[] applicantListField;
        
        private string finalCreditBureauScoreField;
        
        /// <remarks/>
        public string applicationId {
            get {
                return this.applicationIdField;
            }
            set {
                this.applicationIdField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime applicationDate {
            get {
                return this.applicationDateField;
            }
            set {
                this.applicationDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool applicationDateSpecified {
            get {
                return this.applicationDateFieldSpecified;
            }
            set {
                this.applicationDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public ApplicationStatus applicationStatus {
            get {
                return this.applicationStatusField;
            }
            set {
                this.applicationStatusField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool applicationStatusSpecified {
            get {
                return this.applicationStatusFieldSpecified;
            }
            set {
                this.applicationStatusFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("productAppliedFor", IsNullable=false)]
        public ProductAppliedFor[] productAppliedForList {
            get {
                return this.productAppliedForListField;
            }
            set {
                this.productAppliedForListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("applicant", IsNullable=false)]
        public Applicant[] applicantList {
            get {
                return this.applicantListField;
            }
            set {
                this.applicantListField = value;
            }
        }
        
        /// <remarks/>
        public string finalCreditBureauScore {
            get {
                return this.finalCreditBureauScoreField;
            }
            set {
                this.finalCreditBureauScoreField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Application.xsd")]
    public enum ApplicationStatus {
        
        /// <remarks/>
        New,
        
        /// <remarks/>
        PreApproved,
        
        /// <remarks/>
        Assigned,
        
        /// <remarks/>
        InDiscussion,
        
        /// <remarks/>
        Approved,
        
        /// <remarks/>
        Declined,
        
        /// <remarks/>
        Converting,
        
        /// <remarks/>
        Booked,
        
        /// <remarks/>
        Lost,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Application.xsd")]
    public partial class ProductAppliedFor {
        
        private string productAppliedForIdField;
        
        private AccountType productTypeField;
        
        private bool productTypeFieldSpecified;
        
        private string productSubTypeField;
        
        private ApplicationStatus productApplicationStatusField;
        
        private ProductDetail productDetailField;
        
        /// <remarks/>
        public string productAppliedForId {
            get {
                return this.productAppliedForIdField;
            }
            set {
                this.productAppliedForIdField = value;
            }
        }
        
        /// <remarks/>
        public AccountType productType {
            get {
                return this.productTypeField;
            }
            set {
                this.productTypeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool productTypeSpecified {
            get {
                return this.productTypeFieldSpecified;
            }
            set {
                this.productTypeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string productSubType {
            get {
                return this.productSubTypeField;
            }
            set {
                this.productSubTypeField = value;
            }
        }
        
        /// <remarks/>
        public ApplicationStatus productApplicationStatus {
            get {
                return this.productApplicationStatusField;
            }
            set {
                this.productApplicationStatusField = value;
            }
        }
        
        /// <remarks/>
        public ProductDetail productDetail {
            get {
                return this.productDetailField;
            }
            set {
                this.productDetailField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Application.xsd")]
    public partial class ProductDetail {
        
        private Account itemField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("deposit", typeof(Deposit))]
        [System.Xml.Serialization.XmlElementAttribute("loan", typeof(Loan))]
        public Account Item {
            get {
                return this.itemField;
            }
            set {
                this.itemField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Application.xsd")]
    public partial class Applicant {
        
        private string applicantIdField;
        
        private ApplicantRole roleField;
        
        private CreditReport[] creditReportListField;
        
        private string relationshipIdField;
        
        private string partyIdField;
        
        private Party partyField;
        
        /// <remarks/>
        public string applicantId {
            get {
                return this.applicantIdField;
            }
            set {
                this.applicantIdField = value;
            }
        }
        
        /// <remarks/>
        public ApplicantRole role {
            get {
                return this.roleField;
            }
            set {
                this.roleField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("creditReport", Namespace="http://cufxstandards.com/v3/CreditReport.xsd", IsNullable=false)]
        public CreditReport[] creditReportList {
            get {
                return this.creditReportListField;
            }
            set {
                this.creditReportListField = value;
            }
        }
        
        /// <remarks/>
        public string relationshipId {
            get {
                return this.relationshipIdField;
            }
            set {
                this.relationshipIdField = value;
            }
        }
        
        /// <remarks/>
        public string partyId {
            get {
                return this.partyIdField;
            }
            set {
                this.partyIdField = value;
            }
        }
        
        /// <remarks/>
        public Party party {
            get {
                return this.partyField;
            }
            set {
                this.partyField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(AnonymousType=true, Namespace="http://cufxstandards.com/v3/Application.xsd")]
    public enum ApplicantRole {
        
        /// <remarks/>
        Primary,
        
        /// <remarks/>
        Secondary,
        
        /// <remarks/>
        AuthorizedUser,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/ApplicationFilter.xsd")]
    public partial class ApplicationFilter {
        
        private string[] applicationIdListField;
        
        private AccountType[] productTypeListField;
        
        private string productSubTypeListField;
        
        private string[] relationshipIdListField;
        
        private string[] partyIdListField;
        
        private System.DateTime applicationStartDateTimeField;
        
        private bool applicationStartDateTimeFieldSpecified;
        
        private System.DateTime applicationEndDateTimeField;
        
        private bool applicationEndDateTimeFieldSpecified;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("applicationId", Namespace="http://cufxstandards.com/v3/Application.xsd", IsNullable=false)]
        public string[] applicationIdList {
            get {
                return this.applicationIdListField;
            }
            set {
                this.applicationIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("productType", Namespace="http://cufxstandards.com/v3/ProductOffering.xsd", IsNullable=false)]
        public AccountType[] productTypeList {
            get {
                return this.productTypeListField;
            }
            set {
                this.productTypeListField = value;
            }
        }
        
        /// <remarks/>
        public string productSubTypeList {
            get {
                return this.productSubTypeListField;
            }
            set {
                this.productSubTypeListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("relationshipId", Namespace="http://cufxstandards.com/v3/Relationship.xsd", IsNullable=false)]
        public string[] relationshipIdList {
            get {
                return this.relationshipIdListField;
            }
            set {
                this.relationshipIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("partyId", Namespace="http://cufxstandards.com/v3/Party.xsd", IsNullable=false)]
        public string[] partyIdList {
            get {
                return this.partyIdListField;
            }
            set {
                this.partyIdListField = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime applicationStartDateTime {
            get {
                return this.applicationStartDateTimeField;
            }
            set {
                this.applicationStartDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool applicationStartDateTimeSpecified {
            get {
                return this.applicationStartDateTimeFieldSpecified;
            }
            set {
                this.applicationStartDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime applicationEndDateTime {
            get {
                return this.applicationEndDateTimeField;
            }
            set {
                this.applicationEndDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool applicationEndDateTimeSpecified {
            get {
                return this.applicationEndDateTimeFieldSpecified;
            }
            set {
                this.applicationEndDateTimeFieldSpecified = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/ApplicationMessage.xsd")]
    public partial class ApplicationMessage {
        
        private MessageContext messageContextField;
        
        private ApplicationFilter applicationFilterField;
        
        private Application[] applicationListField;
        
        /// <remarks/>
        public MessageContext messageContext {
            get {
                return this.messageContextField;
            }
            set {
                this.messageContextField = value;
            }
        }
        
        /// <remarks/>
        public ApplicationFilter applicationFilter {
            get {
                return this.applicationFilterField;
            }
            set {
                this.applicationFilterField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("application", Namespace="http://cufxstandards.com/v3/Application.xsd", IsNullable=false)]
        public Application[] applicationList {
            get {
                return this.applicationListField;
            }
            set {
                this.applicationListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Artifact.xsd")]
    public partial class ArtifactList {
        
        private Artifact[] artifactField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("artifact")]
        public Artifact[] artifact {
            get {
                return this.artifactField;
            }
            set {
                this.artifactField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Artifact.xsd")]
    public partial class Artifact {
        
        private ArtifactId artifactIdField;
        
        private string artifactTypeField;
        
        private string externalIdField;
        
        private byte[] artifactField;
        
        private string artifactNameField;
        
        private string artifactDescriptionField;
        
        private string artifactOwnerField;
        
        private System.DateTime artifactCreationDateField;
        
        private bool artifactCreationDateFieldSpecified;
        
        private System.DateTime artifactModifiedDateField;
        
        private bool artifactModifiedDateFieldSpecified;
        
        private System.DateTime artifactArchivedDateField;
        
        private bool artifactArchivedDateFieldSpecified;
        
        private System.DateTime artifactDeletedDateField;
        
        private bool artifactDeletedDateFieldSpecified;
        
        private string artifactCompressionTypeField;
        
        private bool artifactArchivedField;
        
        private bool artifactArchivedFieldSpecified;
        
        private ValuePair[] customDataField;
        
        /// <remarks/>
        public ArtifactId artifactId {
            get {
                return this.artifactIdField;
            }
            set {
                this.artifactIdField = value;
            }
        }
        
        /// <remarks/>
        public string artifactType {
            get {
                return this.artifactTypeField;
            }
            set {
                this.artifactTypeField = value;
            }
        }
        
        /// <remarks/>
        public string externalId {
            get {
                return this.externalIdField;
            }
            set {
                this.externalIdField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="base64Binary")]
        public byte[] artifact {
            get {
                return this.artifactField;
            }
            set {
                this.artifactField = value;
            }
        }
        
        /// <remarks/>
        public string artifactName {
            get {
                return this.artifactNameField;
            }
            set {
                this.artifactNameField = value;
            }
        }
        
        /// <remarks/>
        public string artifactDescription {
            get {
                return this.artifactDescriptionField;
            }
            set {
                this.artifactDescriptionField = value;
            }
        }
        
        /// <remarks/>
        public string artifactOwner {
            get {
                return this.artifactOwnerField;
            }
            set {
                this.artifactOwnerField = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime artifactCreationDate {
            get {
                return this.artifactCreationDateField;
            }
            set {
                this.artifactCreationDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool artifactCreationDateSpecified {
            get {
                return this.artifactCreationDateFieldSpecified;
            }
            set {
                this.artifactCreationDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime artifactModifiedDate {
            get {
                return this.artifactModifiedDateField;
            }
            set {
                this.artifactModifiedDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool artifactModifiedDateSpecified {
            get {
                return this.artifactModifiedDateFieldSpecified;
            }
            set {
                this.artifactModifiedDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime artifactArchivedDate {
            get {
                return this.artifactArchivedDateField;
            }
            set {
                this.artifactArchivedDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool artifactArchivedDateSpecified {
            get {
                return this.artifactArchivedDateFieldSpecified;
            }
            set {
                this.artifactArchivedDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime artifactDeletedDate {
            get {
                return this.artifactDeletedDateField;
            }
            set {
                this.artifactDeletedDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool artifactDeletedDateSpecified {
            get {
                return this.artifactDeletedDateFieldSpecified;
            }
            set {
                this.artifactDeletedDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string artifactCompressionType {
            get {
                return this.artifactCompressionTypeField;
            }
            set {
                this.artifactCompressionTypeField = value;
            }
        }
        
        /// <remarks/>
        public bool artifactArchived {
            get {
                return this.artifactArchivedField;
            }
            set {
                this.artifactArchivedField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool artifactArchivedSpecified {
            get {
                return this.artifactArchivedFieldSpecified;
            }
            set {
                this.artifactArchivedFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("valuePair", Namespace="http://cufxstandards.com/v3/Common.xsd", IsNullable=false)]
        public ValuePair[] customData {
            get {
                return this.customDataField;
            }
            set {
                this.customDataField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Artifact.xsd")]
    public partial class ArtifactId {
        
        private object itemField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("artifactIdKeyValueList", typeof(CustomData))]
        [System.Xml.Serialization.XmlElementAttribute("artifactUniqueId", typeof(string))]
        public object Item {
            get {
                return this.itemField;
            }
            set {
                this.itemField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Common.xsd")]
    public partial class CustomData {
        
        private ValuePair[] valuePairField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("valuePair")]
        public ValuePair[] valuePair {
            get {
                return this.valuePairField;
            }
            set {
                this.valuePairField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/ArtifactFilter.xsd")]
    public partial class ArtifactFilter {
        
        private ArtifactId[] artifactIdListField;
        
        private string artifactNameField;
        
        private string artifactDescriptionField;
        
        private bool artifactArchivedField;
        
        private bool artifactArchivedFieldSpecified;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("artifactId", Namespace="http://cufxstandards.com/v3/Artifact.xsd")]
        public ArtifactId[] artifactIdList {
            get {
                return this.artifactIdListField;
            }
            set {
                this.artifactIdListField = value;
            }
        }
        
        /// <remarks/>
        public string artifactName {
            get {
                return this.artifactNameField;
            }
            set {
                this.artifactNameField = value;
            }
        }
        
        /// <remarks/>
        public string artifactDescription {
            get {
                return this.artifactDescriptionField;
            }
            set {
                this.artifactDescriptionField = value;
            }
        }
        
        /// <remarks/>
        public bool artifactArchived {
            get {
                return this.artifactArchivedField;
            }
            set {
                this.artifactArchivedField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool artifactArchivedSpecified {
            get {
                return this.artifactArchivedFieldSpecified;
            }
            set {
                this.artifactArchivedFieldSpecified = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/ArtifactMessage.xsd")]
    public partial class ArtifactMessage {
        
        private MessageContext messageContextField;
        
        private ArtifactFilter artifactFilterField;
        
        private Artifact[] artifactListField;
        
        /// <remarks/>
        public MessageContext messageContext {
            get {
                return this.messageContextField;
            }
            set {
                this.messageContextField = value;
            }
        }
        
        /// <remarks/>
        public ArtifactFilter artifactFilter {
            get {
                return this.artifactFilterField;
            }
            set {
                this.artifactFilterField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("artifact", Namespace="http://cufxstandards.com/v3/Artifact.xsd", IsNullable=false)]
        public Artifact[] artifactList {
            get {
                return this.artifactListField;
            }
            set {
                this.artifactListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Bill.xsd")]
    public partial class BillList {
        
        private Bill[] billField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("bill")]
        public Bill[] bill {
            get {
                return this.billField;
            }
            set {
                this.billField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Bill.xsd")]
    public partial class Bill {
        
        private string billIdField;
        
        private ArtifactId billImageArtifactIdField;
        
        private string eBillUriField;
        
        private string billFromPayeeIdField;
        
        private string billFromPayeeGlobalIdField;
        
        private string partyIdField;
        
        private string relationshipIdField;
        
        private string accountIdField;
        
        private bool isEBillField;
        
        private Money dueAmountField;
        
        private Money earlyPaymentAmountField;
        
        private Money minimumPaymentAmountField;
        
        private System.DateTime sentDateTimeField;
        
        private bool sentDateTimeFieldSpecified;
        
        private System.DateTime dueDateTimeField;
        
        private bool dueDateTimeFieldSpecified;
        
        private System.DateTime earlyPaymentDateTimeField;
        
        private bool earlyPaymentDateTimeFieldSpecified;
        
        private Money lateFeeField;
        
        private System.DateTime updatedDateTimeField;
        
        private bool updatedDateTimeFieldSpecified;
        
        private ValuePair[] customDataField;
        
        /// <remarks/>
        public string billId {
            get {
                return this.billIdField;
            }
            set {
                this.billIdField = value;
            }
        }
        
        /// <remarks/>
        public ArtifactId billImageArtifactId {
            get {
                return this.billImageArtifactIdField;
            }
            set {
                this.billImageArtifactIdField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="anyURI")]
        public string eBillUri {
            get {
                return this.eBillUriField;
            }
            set {
                this.eBillUriField = value;
            }
        }
        
        /// <remarks/>
        public string billFromPayeeId {
            get {
                return this.billFromPayeeIdField;
            }
            set {
                this.billFromPayeeIdField = value;
            }
        }
        
        /// <remarks/>
        public string billFromPayeeGlobalId {
            get {
                return this.billFromPayeeGlobalIdField;
            }
            set {
                this.billFromPayeeGlobalIdField = value;
            }
        }
        
        /// <remarks/>
        public string partyId {
            get {
                return this.partyIdField;
            }
            set {
                this.partyIdField = value;
            }
        }
        
        /// <remarks/>
        public string relationshipId {
            get {
                return this.relationshipIdField;
            }
            set {
                this.relationshipIdField = value;
            }
        }
        
        /// <remarks/>
        public string accountId {
            get {
                return this.accountIdField;
            }
            set {
                this.accountIdField = value;
            }
        }
        
        /// <remarks/>
        public bool isEBill {
            get {
                return this.isEBillField;
            }
            set {
                this.isEBillField = value;
            }
        }
        
        /// <remarks/>
        public Money dueAmount {
            get {
                return this.dueAmountField;
            }
            set {
                this.dueAmountField = value;
            }
        }
        
        /// <remarks/>
        public Money earlyPaymentAmount {
            get {
                return this.earlyPaymentAmountField;
            }
            set {
                this.earlyPaymentAmountField = value;
            }
        }
        
        /// <remarks/>
        public Money minimumPaymentAmount {
            get {
                return this.minimumPaymentAmountField;
            }
            set {
                this.minimumPaymentAmountField = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime sentDateTime {
            get {
                return this.sentDateTimeField;
            }
            set {
                this.sentDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool sentDateTimeSpecified {
            get {
                return this.sentDateTimeFieldSpecified;
            }
            set {
                this.sentDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime dueDateTime {
            get {
                return this.dueDateTimeField;
            }
            set {
                this.dueDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool dueDateTimeSpecified {
            get {
                return this.dueDateTimeFieldSpecified;
            }
            set {
                this.dueDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime earlyPaymentDateTime {
            get {
                return this.earlyPaymentDateTimeField;
            }
            set {
                this.earlyPaymentDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool earlyPaymentDateTimeSpecified {
            get {
                return this.earlyPaymentDateTimeFieldSpecified;
            }
            set {
                this.earlyPaymentDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public Money lateFee {
            get {
                return this.lateFeeField;
            }
            set {
                this.lateFeeField = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime updatedDateTime {
            get {
                return this.updatedDateTimeField;
            }
            set {
                this.updatedDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool updatedDateTimeSpecified {
            get {
                return this.updatedDateTimeFieldSpecified;
            }
            set {
                this.updatedDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("valuePair", Namespace="http://cufxstandards.com/v3/Common.xsd", IsNullable=false)]
        public ValuePair[] customData {
            get {
                return this.customDataField;
            }
            set {
                this.customDataField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/BillFilter.xsd")]
    public partial class BillFilter {
        
        private string[] billIdListField;
        
        private string[] partyIdListField;
        
        private string[] relationshipIdListField;
        
        private string[] accountIdListField;
        
        private string[] billFromPayeeIdListField;
        
        private bool isEBillField;
        
        private System.DateTime billSentStartDateTimeField;
        
        private bool billSentStartDateTimeFieldSpecified;
        
        private System.DateTime billSentEndDateTimeField;
        
        private bool billSentEndDateTimeFieldSpecified;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("billId", Namespace="http://cufxstandards.com/v3/Bill.xsd", IsNullable=false)]
        public string[] billIdList {
            get {
                return this.billIdListField;
            }
            set {
                this.billIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("partyId", Namespace="http://cufxstandards.com/v3/Party.xsd", IsNullable=false)]
        public string[] partyIdList {
            get {
                return this.partyIdListField;
            }
            set {
                this.partyIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("relationshipId", Namespace="http://cufxstandards.com/v3/Relationship.xsd", IsNullable=false)]
        public string[] relationshipIdList {
            get {
                return this.relationshipIdListField;
            }
            set {
                this.relationshipIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("accountId", Namespace="http://cufxstandards.com/v3/Account.xsd", IsNullable=false)]
        public string[] accountIdList {
            get {
                return this.accountIdListField;
            }
            set {
                this.accountIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("billPayeeId", Namespace="http://cufxstandards.com/v3/BillPayee.xsd", IsNullable=false)]
        public string[] billFromPayeeIdList {
            get {
                return this.billFromPayeeIdListField;
            }
            set {
                this.billFromPayeeIdListField = value;
            }
        }
        
        /// <remarks/>
        public bool isEBill {
            get {
                return this.isEBillField;
            }
            set {
                this.isEBillField = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime billSentStartDateTime {
            get {
                return this.billSentStartDateTimeField;
            }
            set {
                this.billSentStartDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool billSentStartDateTimeSpecified {
            get {
                return this.billSentStartDateTimeFieldSpecified;
            }
            set {
                this.billSentStartDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime billSentEndDateTime {
            get {
                return this.billSentEndDateTimeField;
            }
            set {
                this.billSentEndDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool billSentEndDateTimeSpecified {
            get {
                return this.billSentEndDateTimeFieldSpecified;
            }
            set {
                this.billSentEndDateTimeFieldSpecified = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/BillMessage.xsd")]
    public partial class BillMessage {
        
        private MessageContext messageContextField;
        
        private BillFilter billFilterField;
        
        private Bill[] billListField;
        
        /// <remarks/>
        public MessageContext messageContext {
            get {
                return this.messageContextField;
            }
            set {
                this.messageContextField = value;
            }
        }
        
        /// <remarks/>
        public BillFilter billFilter {
            get {
                return this.billFilterField;
            }
            set {
                this.billFilterField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("bill", Namespace="http://cufxstandards.com/v3/Bill.xsd", IsNullable=false)]
        public Bill[] billList {
            get {
                return this.billListField;
            }
            set {
                this.billListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/BillPayee.xsd")]
    public partial class BillPayeeList {
        
        private BillPayee[] billPayeeField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("billPayee")]
        public BillPayee[] billPayee {
            get {
                return this.billPayeeField;
            }
            set {
                this.billPayeeField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/BillPayee.xsd")]
    public partial class BillPayee {
        
        private string billPayeeIdField;
        
        private string billPayeeGlobalIdField;
        
        private string payeeNameField;
        
        private ContactAddress payeeAddressField;
        
        private Phone payeePhoneField;
        
        private Email payeeEmailField;
        
        private string processorNameField;
        
        private string payeeACHRoutingNumberField;
        
        private string payeeACHAccountNumberField;
        
        private bool isElectronicField;
        
        private bool isEBillProviderField;
        
        private bool isEBillEnrolledField;
        
        private System.DateTime addedDateTimeField;
        
        private bool addedDateTimeFieldSpecified;
        
        private System.DateTime updatedDateTimeField;
        
        private bool updatedDateTimeFieldSpecified;
        
        private int minimumDaysToPayField;
        
        private string partyIdField;
        
        private string relationshipIdField;
        
        private string accountIdField;
        
        private string userDefinedNameField;
        
        private string payeeAccountNumberField;
        
        private string accountHolderNameField;
        
        private string payeeCategoryField;
        
        private PayeeStatus payeeStatusField;
        
        private bool payeeStatusFieldSpecified;
        
        private string defaultPaymentFromAccountIdField;
        
        private ValuePair[] customDataField;
        
        /// <remarks/>
        public string billPayeeId {
            get {
                return this.billPayeeIdField;
            }
            set {
                this.billPayeeIdField = value;
            }
        }
        
        /// <remarks/>
        public string billPayeeGlobalId {
            get {
                return this.billPayeeGlobalIdField;
            }
            set {
                this.billPayeeGlobalIdField = value;
            }
        }
        
        /// <remarks/>
        public string payeeName {
            get {
                return this.payeeNameField;
            }
            set {
                this.payeeNameField = value;
            }
        }
        
        /// <remarks/>
        public ContactAddress payeeAddress {
            get {
                return this.payeeAddressField;
            }
            set {
                this.payeeAddressField = value;
            }
        }
        
        /// <remarks/>
        public Phone payeePhone {
            get {
                return this.payeePhoneField;
            }
            set {
                this.payeePhoneField = value;
            }
        }
        
        /// <remarks/>
        public Email payeeEmail {
            get {
                return this.payeeEmailField;
            }
            set {
                this.payeeEmailField = value;
            }
        }
        
        /// <remarks/>
        public string processorName {
            get {
                return this.processorNameField;
            }
            set {
                this.processorNameField = value;
            }
        }
        
        /// <remarks/>
        public string payeeACHRoutingNumber {
            get {
                return this.payeeACHRoutingNumberField;
            }
            set {
                this.payeeACHRoutingNumberField = value;
            }
        }
        
        /// <remarks/>
        public string payeeACHAccountNumber {
            get {
                return this.payeeACHAccountNumberField;
            }
            set {
                this.payeeACHAccountNumberField = value;
            }
        }
        
        /// <remarks/>
        public bool isElectronic {
            get {
                return this.isElectronicField;
            }
            set {
                this.isElectronicField = value;
            }
        }
        
        /// <remarks/>
        public bool isEBillProvider {
            get {
                return this.isEBillProviderField;
            }
            set {
                this.isEBillProviderField = value;
            }
        }
        
        /// <remarks/>
        public bool isEBillEnrolled {
            get {
                return this.isEBillEnrolledField;
            }
            set {
                this.isEBillEnrolledField = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime addedDateTime {
            get {
                return this.addedDateTimeField;
            }
            set {
                this.addedDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool addedDateTimeSpecified {
            get {
                return this.addedDateTimeFieldSpecified;
            }
            set {
                this.addedDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime updatedDateTime {
            get {
                return this.updatedDateTimeField;
            }
            set {
                this.updatedDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool updatedDateTimeSpecified {
            get {
                return this.updatedDateTimeFieldSpecified;
            }
            set {
                this.updatedDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public int minimumDaysToPay {
            get {
                return this.minimumDaysToPayField;
            }
            set {
                this.minimumDaysToPayField = value;
            }
        }
        
        /// <remarks/>
        public string partyId {
            get {
                return this.partyIdField;
            }
            set {
                this.partyIdField = value;
            }
        }
        
        /// <remarks/>
        public string relationshipId {
            get {
                return this.relationshipIdField;
            }
            set {
                this.relationshipIdField = value;
            }
        }
        
        /// <remarks/>
        public string accountId {
            get {
                return this.accountIdField;
            }
            set {
                this.accountIdField = value;
            }
        }
        
        /// <remarks/>
        public string userDefinedName {
            get {
                return this.userDefinedNameField;
            }
            set {
                this.userDefinedNameField = value;
            }
        }
        
        /// <remarks/>
        public string payeeAccountNumber {
            get {
                return this.payeeAccountNumberField;
            }
            set {
                this.payeeAccountNumberField = value;
            }
        }
        
        /// <remarks/>
        public string accountHolderName {
            get {
                return this.accountHolderNameField;
            }
            set {
                this.accountHolderNameField = value;
            }
        }
        
        /// <remarks/>
        public string payeeCategory {
            get {
                return this.payeeCategoryField;
            }
            set {
                this.payeeCategoryField = value;
            }
        }
        
        /// <remarks/>
        public PayeeStatus payeeStatus {
            get {
                return this.payeeStatusField;
            }
            set {
                this.payeeStatusField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool payeeStatusSpecified {
            get {
                return this.payeeStatusFieldSpecified;
            }
            set {
                this.payeeStatusFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string defaultPaymentFromAccountId {
            get {
                return this.defaultPaymentFromAccountIdField;
            }
            set {
                this.defaultPaymentFromAccountIdField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("valuePair", Namespace="http://cufxstandards.com/v3/Common.xsd", IsNullable=false)]
        public ValuePair[] customData {
            get {
                return this.customDataField;
            }
            set {
                this.customDataField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/BillPayee.xsd")]
    public enum PayeeStatus {
        
        /// <remarks/>
        Active,
        
        /// <remarks/>
        Inactive,
        
        /// <remarks/>
        Deleted,
        
        /// <remarks/>
        Invalid,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/BillPayeeFilter.xsd")]
    public partial class BillPayeeFilter {
        
        private string[] billPayeeGlobalIdListField;
        
        private string[] partyIdListField;
        
        private string[] relationshipIdListField;
        
        private string[] accountIdListField;
        
        private string payeeNameField;
        
        private bool isElectronicField;
        
        private string userDefinedNameField;
        
        private string payeeCategoryField;
        
        private PayeeStatus[] payeeStatusListField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("billPayeeGlobalId", Namespace="http://cufxstandards.com/v3/BillPayee.xsd", IsNullable=false)]
        public string[] billPayeeGlobalIdList {
            get {
                return this.billPayeeGlobalIdListField;
            }
            set {
                this.billPayeeGlobalIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("partyId", Namespace="http://cufxstandards.com/v3/Party.xsd", IsNullable=false)]
        public string[] partyIdList {
            get {
                return this.partyIdListField;
            }
            set {
                this.partyIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("relationshipId", Namespace="http://cufxstandards.com/v3/Relationship.xsd", IsNullable=false)]
        public string[] relationshipIdList {
            get {
                return this.relationshipIdListField;
            }
            set {
                this.relationshipIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("accountId", Namespace="http://cufxstandards.com/v3/Account.xsd", IsNullable=false)]
        public string[] accountIdList {
            get {
                return this.accountIdListField;
            }
            set {
                this.accountIdListField = value;
            }
        }
        
        /// <remarks/>
        public string payeeName {
            get {
                return this.payeeNameField;
            }
            set {
                this.payeeNameField = value;
            }
        }
        
        /// <remarks/>
        public bool isElectronic {
            get {
                return this.isElectronicField;
            }
            set {
                this.isElectronicField = value;
            }
        }
        
        /// <remarks/>
        public string userDefinedName {
            get {
                return this.userDefinedNameField;
            }
            set {
                this.userDefinedNameField = value;
            }
        }
        
        /// <remarks/>
        public string payeeCategory {
            get {
                return this.payeeCategoryField;
            }
            set {
                this.payeeCategoryField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("payeeStatus", Namespace="http://cufxstandards.com/v3/BillPayee.xsd", IsNullable=false)]
        public PayeeStatus[] payeeStatusList {
            get {
                return this.payeeStatusListField;
            }
            set {
                this.payeeStatusListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/BillPayeeMessage.xsd")]
    public partial class BillPayeeMessage {
        
        private MessageContext messageContextField;
        
        private BillPayeeFilter billPayeeFilterField;
        
        private BillPayee[] billPayeeListField;
        
        /// <remarks/>
        public MessageContext messageContext {
            get {
                return this.messageContextField;
            }
            set {
                this.messageContextField = value;
            }
        }
        
        /// <remarks/>
        public BillPayeeFilter billPayeeFilter {
            get {
                return this.billPayeeFilterField;
            }
            set {
                this.billPayeeFilterField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("billPayee", Namespace="http://cufxstandards.com/v3/BillPayee.xsd", IsNullable=false)]
        public BillPayee[] billPayeeList {
            get {
                return this.billPayeeListField;
            }
            set {
                this.billPayeeListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/BillPaymentFilter.xsd")]
    public partial class BillPaymentFilter : FundsTransferFilterBase {
        
        private string[] billPayeeIdListField;
        
        private string[] billIdListField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("billPayeeId", Namespace="http://cufxstandards.com/v3/BillPayee.xsd", IsNullable=false)]
        public string[] billPayeeIdList {
            get {
                return this.billPayeeIdListField;
            }
            set {
                this.billPayeeIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("billId", Namespace="http://cufxstandards.com/v3/Bill.xsd", IsNullable=false)]
        public string[] billIdList {
            get {
                return this.billIdListField;
            }
            set {
                this.billIdListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/FundsTransferFilterBase.xsd")]
    public abstract partial class FundsTransferFilterBase {
        
        private string[] occurrenceIdListField;
        
        private string[] recurringIdListField;
        
        private System.DateTime startCompletedDateTimeField;
        
        private bool startCompletedDateTimeFieldSpecified;
        
        private System.DateTime endCompletedDateTimeField;
        
        private bool endCompletedDateTimeFieldSpecified;
        
        private string[] accountIDListField;
        
        private string[] partyIdListField;
        
        private string[] relationshipIdListField;
        
        private OccurrenceStatus[] occurrenceStatusListField;
        
        private Money minAmountField;
        
        private Money maxAmountField;
        
        private string[] batchIdListField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("occurrenceId", Namespace="http://cufxstandards.com/v3/FundsTransferCommonBase.xsd", IsNullable=false)]
        public string[] occurrenceIdList {
            get {
                return this.occurrenceIdListField;
            }
            set {
                this.occurrenceIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("recurringId", Namespace="http://cufxstandards.com/v3/FundsTransferCommonBase.xsd", IsNullable=false)]
        public string[] recurringIdList {
            get {
                return this.recurringIdListField;
            }
            set {
                this.recurringIdListField = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime startCompletedDateTime {
            get {
                return this.startCompletedDateTimeField;
            }
            set {
                this.startCompletedDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool startCompletedDateTimeSpecified {
            get {
                return this.startCompletedDateTimeFieldSpecified;
            }
            set {
                this.startCompletedDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime endCompletedDateTime {
            get {
                return this.endCompletedDateTimeField;
            }
            set {
                this.endCompletedDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool endCompletedDateTimeSpecified {
            get {
                return this.endCompletedDateTimeFieldSpecified;
            }
            set {
                this.endCompletedDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("accountId", Namespace="http://cufxstandards.com/v3/Account.xsd", IsNullable=false)]
        public string[] accountIDList {
            get {
                return this.accountIDListField;
            }
            set {
                this.accountIDListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("partyId", Namespace="http://cufxstandards.com/v3/Party.xsd", IsNullable=false)]
        public string[] partyIdList {
            get {
                return this.partyIdListField;
            }
            set {
                this.partyIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("relationshipId", Namespace="http://cufxstandards.com/v3/Relationship.xsd", IsNullable=false)]
        public string[] relationshipIdList {
            get {
                return this.relationshipIdListField;
            }
            set {
                this.relationshipIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("occurrenceStatus", Namespace="http://cufxstandards.com/v3/FundsTransferCommonBase.xsd", IsNullable=false)]
        public OccurrenceStatus[] occurrenceStatusList {
            get {
                return this.occurrenceStatusListField;
            }
            set {
                this.occurrenceStatusListField = value;
            }
        }
        
        /// <remarks/>
        public Money minAmount {
            get {
                return this.minAmountField;
            }
            set {
                this.minAmountField = value;
            }
        }
        
        /// <remarks/>
        public Money maxAmount {
            get {
                return this.maxAmountField;
            }
            set {
                this.maxAmountField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("batchId", Namespace="http://cufxstandards.com/v3/FundsTransferCommonBase.xsd", IsNullable=false)]
        public string[] batchIdList {
            get {
                return this.batchIdListField;
            }
            set {
                this.batchIdListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/FundsTransferCommonBase.xsd")]
    public enum OccurrenceStatus {
        
        /// <remarks/>
        Scheduled,
        
        /// <remarks/>
        InProcess,
        
        /// <remarks/>
        CancelRequestedByParty,
        
        /// <remarks/>
        CancelRequestedByFinancialInstitution,
        
        /// <remarks/>
        CancelRequestedByProcessor,
        
        /// <remarks/>
        Cancelled,
        
        /// <remarks/>
        Sent,
        
        /// <remarks/>
        Completed,
        
        /// <remarks/>
        Declined,
        
        /// <remarks/>
        FundsOutbound,
        
        /// <remarks/>
        FundsCleared,
        
        /// <remarks/>
        Held,
        
        /// <remarks/>
        InsufficientFunds,
        
        /// <remarks/>
        Returned,
        
        /// <remarks/>
        Suspended,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/WireFilter.xsd")]
    public partial class WireFilter : FundsTransferFilterBase {
        
        private WireTransferType[] wireTransferTypeListField;
        
        private string escrowNumberField;
        
        private string escrowOfficerNameField;
        
        private string micrAccountNumberField;
        
        private string routingNumberField;
        
        private string externalAccountSWIFTCodeField;
        
        private string externalAccountIBANCodeField;
        
        private string externalAccountBankCodeField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("wireTransferType", Namespace="http://cufxstandards.com/v3/Wire.xsd", IsNullable=false)]
        public WireTransferType[] wireTransferTypeList {
            get {
                return this.wireTransferTypeListField;
            }
            set {
                this.wireTransferTypeListField = value;
            }
        }
        
        /// <remarks/>
        public string escrowNumber {
            get {
                return this.escrowNumberField;
            }
            set {
                this.escrowNumberField = value;
            }
        }
        
        /// <remarks/>
        public string escrowOfficerName {
            get {
                return this.escrowOfficerNameField;
            }
            set {
                this.escrowOfficerNameField = value;
            }
        }
        
        /// <remarks/>
        public string micrAccountNumber {
            get {
                return this.micrAccountNumberField;
            }
            set {
                this.micrAccountNumberField = value;
            }
        }
        
        /// <remarks/>
        public string routingNumber {
            get {
                return this.routingNumberField;
            }
            set {
                this.routingNumberField = value;
            }
        }
        
        /// <remarks/>
        public string externalAccountSWIFTCode {
            get {
                return this.externalAccountSWIFTCodeField;
            }
            set {
                this.externalAccountSWIFTCodeField = value;
            }
        }
        
        /// <remarks/>
        public string externalAccountIBANCode {
            get {
                return this.externalAccountIBANCodeField;
            }
            set {
                this.externalAccountIBANCodeField = value;
            }
        }
        
        /// <remarks/>
        public string externalAccountBankCode {
            get {
                return this.externalAccountBankCodeField;
            }
            set {
                this.externalAccountBankCodeField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Wire.xsd")]
    public enum WireTransferType {
        
        /// <remarks/>
        Domestic,
        
        /// <remarks/>
        Foreign,
        
        /// <remarks/>
        Investment,
        
        /// <remarks/>
        Escrow,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/TransferFilter.xsd")]
    public partial class TransferFilter : FundsTransferFilterBase {
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/BillPaymentOccurrence.xsd")]
    public partial class BillPaymentOccurrenceList {
        
        private BillPaymentOccurrence[] billPaymentOccurrenceField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("billPaymentOccurrence")]
        public BillPaymentOccurrence[] billPaymentOccurrence {
            get {
                return this.billPaymentOccurrenceField;
            }
            set {
                this.billPaymentOccurrenceField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/BillPaymentOccurrence.xsd")]
    public partial class BillPaymentOccurrence : FundsTransferOccurrenceBase {
        
        private string billPayeeIdField;
        
        private string billIdField;
        
        private System.DateTime expectedDebitDateTimeField;
        
        private bool expectedDebitDateTimeFieldSpecified;
        
        private System.DateTime paymentClearedDateTimeField;
        
        private bool paymentClearedDateTimeFieldSpecified;
        
        private bool isStopRequestedField;
        
        private bool isStopRequestedFieldSpecified;
        
        private System.DateTime checkStoppedDateField;
        
        private bool checkStoppedDateFieldSpecified;
        
        private string nsfCountField;
        
        private string checkNumberField;
        
        /// <remarks/>
        public string billPayeeId {
            get {
                return this.billPayeeIdField;
            }
            set {
                this.billPayeeIdField = value;
            }
        }
        
        /// <remarks/>
        public string billId {
            get {
                return this.billIdField;
            }
            set {
                this.billIdField = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime expectedDebitDateTime {
            get {
                return this.expectedDebitDateTimeField;
            }
            set {
                this.expectedDebitDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool expectedDebitDateTimeSpecified {
            get {
                return this.expectedDebitDateTimeFieldSpecified;
            }
            set {
                this.expectedDebitDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime paymentClearedDateTime {
            get {
                return this.paymentClearedDateTimeField;
            }
            set {
                this.paymentClearedDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool paymentClearedDateTimeSpecified {
            get {
                return this.paymentClearedDateTimeFieldSpecified;
            }
            set {
                this.paymentClearedDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public bool isStopRequested {
            get {
                return this.isStopRequestedField;
            }
            set {
                this.isStopRequestedField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool isStopRequestedSpecified {
            get {
                return this.isStopRequestedFieldSpecified;
            }
            set {
                this.isStopRequestedFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime checkStoppedDate {
            get {
                return this.checkStoppedDateField;
            }
            set {
                this.checkStoppedDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool checkStoppedDateSpecified {
            get {
                return this.checkStoppedDateFieldSpecified;
            }
            set {
                this.checkStoppedDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="integer")]
        public string nsfCount {
            get {
                return this.nsfCountField;
            }
            set {
                this.nsfCountField = value;
            }
        }
        
        /// <remarks/>
        public string checkNumber {
            get {
                return this.checkNumberField;
            }
            set {
                this.checkNumberField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/FundsTransferOccurrenceBase.xsd")]
    public abstract partial class FundsTransferOccurrenceBase {
        
        private string occurrenceIdField;
        
        private string recurringIdField;
        
        private Money occurrenceAmountField;
        
        private string occurrenceFromAccountIdField;
        
        private string occurrenceToAccountIdField;
        
        private OccurrenceStatus occurrenceStatusField;
        
        private bool occurrenceStatusFieldSpecified;
        
        private FundsWithdrawalType fundsWithdrawalTypeField;
        
        private bool fundsWithdrawalTypeFieldSpecified;
        
        private string occurrenceMemoField;
        
        private System.DateTime estimatedProcessDateTimeField;
        
        private bool estimatedProcessDateTimeFieldSpecified;
        
        private System.DateTime createdDateTimeField;
        
        private bool createdDateTimeFieldSpecified;
        
        private System.DateTime processingStartedDateTimeField;
        
        private bool processingStartedDateTimeFieldSpecified;
        
        private System.DateTime processedDateTimeField;
        
        private bool processedDateTimeFieldSpecified;
        
        private System.DateTime completedDateTimeField;
        
        private bool completedDateTimeFieldSpecified;
        
        private System.DateTime lastUpdatedDateTimeField;
        
        private bool lastUpdatedDateTimeFieldSpecified;
        
        private string occurrenceConfirmationCodeField;
        
        private bool queuedForPostingField;
        
        private bool queuedForPostingFieldSpecified;
        
        private string noteField;
        
        private string batchIdField;
        
        private ValuePair[] customDataField;
        
        /// <remarks/>
        public string occurrenceId {
            get {
                return this.occurrenceIdField;
            }
            set {
                this.occurrenceIdField = value;
            }
        }
        
        /// <remarks/>
        public string recurringId {
            get {
                return this.recurringIdField;
            }
            set {
                this.recurringIdField = value;
            }
        }
        
        /// <remarks/>
        public Money occurrenceAmount {
            get {
                return this.occurrenceAmountField;
            }
            set {
                this.occurrenceAmountField = value;
            }
        }
        
        /// <remarks/>
        public string occurrenceFromAccountId {
            get {
                return this.occurrenceFromAccountIdField;
            }
            set {
                this.occurrenceFromAccountIdField = value;
            }
        }
        
        /// <remarks/>
        public string occurrenceToAccountId {
            get {
                return this.occurrenceToAccountIdField;
            }
            set {
                this.occurrenceToAccountIdField = value;
            }
        }
        
        /// <remarks/>
        public OccurrenceStatus occurrenceStatus {
            get {
                return this.occurrenceStatusField;
            }
            set {
                this.occurrenceStatusField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool occurrenceStatusSpecified {
            get {
                return this.occurrenceStatusFieldSpecified;
            }
            set {
                this.occurrenceStatusFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public FundsWithdrawalType fundsWithdrawalType {
            get {
                return this.fundsWithdrawalTypeField;
            }
            set {
                this.fundsWithdrawalTypeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool fundsWithdrawalTypeSpecified {
            get {
                return this.fundsWithdrawalTypeFieldSpecified;
            }
            set {
                this.fundsWithdrawalTypeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string occurrenceMemo {
            get {
                return this.occurrenceMemoField;
            }
            set {
                this.occurrenceMemoField = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime estimatedProcessDateTime {
            get {
                return this.estimatedProcessDateTimeField;
            }
            set {
                this.estimatedProcessDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool estimatedProcessDateTimeSpecified {
            get {
                return this.estimatedProcessDateTimeFieldSpecified;
            }
            set {
                this.estimatedProcessDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime createdDateTime {
            get {
                return this.createdDateTimeField;
            }
            set {
                this.createdDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool createdDateTimeSpecified {
            get {
                return this.createdDateTimeFieldSpecified;
            }
            set {
                this.createdDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime processingStartedDateTime {
            get {
                return this.processingStartedDateTimeField;
            }
            set {
                this.processingStartedDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool processingStartedDateTimeSpecified {
            get {
                return this.processingStartedDateTimeFieldSpecified;
            }
            set {
                this.processingStartedDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime processedDateTime {
            get {
                return this.processedDateTimeField;
            }
            set {
                this.processedDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool processedDateTimeSpecified {
            get {
                return this.processedDateTimeFieldSpecified;
            }
            set {
                this.processedDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime completedDateTime {
            get {
                return this.completedDateTimeField;
            }
            set {
                this.completedDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool completedDateTimeSpecified {
            get {
                return this.completedDateTimeFieldSpecified;
            }
            set {
                this.completedDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime lastUpdatedDateTime {
            get {
                return this.lastUpdatedDateTimeField;
            }
            set {
                this.lastUpdatedDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool lastUpdatedDateTimeSpecified {
            get {
                return this.lastUpdatedDateTimeFieldSpecified;
            }
            set {
                this.lastUpdatedDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string occurrenceConfirmationCode {
            get {
                return this.occurrenceConfirmationCodeField;
            }
            set {
                this.occurrenceConfirmationCodeField = value;
            }
        }
        
        /// <remarks/>
        public bool queuedForPosting {
            get {
                return this.queuedForPostingField;
            }
            set {
                this.queuedForPostingField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool queuedForPostingSpecified {
            get {
                return this.queuedForPostingFieldSpecified;
            }
            set {
                this.queuedForPostingFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string note {
            get {
                return this.noteField;
            }
            set {
                this.noteField = value;
            }
        }
        
        /// <remarks/>
        public string batchId {
            get {
                return this.batchIdField;
            }
            set {
                this.batchIdField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("valuePair", Namespace="http://cufxstandards.com/v3/Common.xsd", IsNullable=false)]
        public ValuePair[] customData {
            get {
                return this.customDataField;
            }
            set {
                this.customDataField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/FundsTransferCommonBase.xsd")]
    public enum FundsWithdrawalType {
        
        /// <remarks/>
        WhenRequested,
        
        /// <remarks/>
        WhenCleared,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Wire.xsd")]
    public partial class Wire : FundsTransferOccurrenceBase {
        
        private WireTransferType wireTransferTypeField;
        
        private bool wireTransferTypeFieldSpecified;
        
        private string wireTransferSpecialInstructionsField;
        
        private string escrowNumberField;
        
        private string escrowOfficerNameField;
        
        private bool disclosuresConsentedField;
        
        private bool disclosuresConsentedFieldSpecified;
        
        private PersonName beneficiaryNameField;
        
        private Address beneficiaryAddressField;
        
        /// <remarks/>
        public WireTransferType wireTransferType {
            get {
                return this.wireTransferTypeField;
            }
            set {
                this.wireTransferTypeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool wireTransferTypeSpecified {
            get {
                return this.wireTransferTypeFieldSpecified;
            }
            set {
                this.wireTransferTypeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string wireTransferSpecialInstructions {
            get {
                return this.wireTransferSpecialInstructionsField;
            }
            set {
                this.wireTransferSpecialInstructionsField = value;
            }
        }
        
        /// <remarks/>
        public string escrowNumber {
            get {
                return this.escrowNumberField;
            }
            set {
                this.escrowNumberField = value;
            }
        }
        
        /// <remarks/>
        public string escrowOfficerName {
            get {
                return this.escrowOfficerNameField;
            }
            set {
                this.escrowOfficerNameField = value;
            }
        }
        
        /// <remarks/>
        public bool disclosuresConsented {
            get {
                return this.disclosuresConsentedField;
            }
            set {
                this.disclosuresConsentedField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool disclosuresConsentedSpecified {
            get {
                return this.disclosuresConsentedFieldSpecified;
            }
            set {
                this.disclosuresConsentedFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public PersonName beneficiaryName {
            get {
                return this.beneficiaryNameField;
            }
            set {
                this.beneficiaryNameField = value;
            }
        }
        
        /// <remarks/>
        public Address beneficiaryAddress {
            get {
                return this.beneficiaryAddressField;
            }
            set {
                this.beneficiaryAddressField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Common.xsd")]
    public partial class PersonName {
        
        private string firstNameField;
        
        private string middleNameField;
        
        private string lastNameField;
        
        private string prefixField;
        
        private string suffixField;
        
        private string formattedNameField;
        
        private string nicknameField;
        
        /// <remarks/>
        public string firstName {
            get {
                return this.firstNameField;
            }
            set {
                this.firstNameField = value;
            }
        }
        
        /// <remarks/>
        public string middleName {
            get {
                return this.middleNameField;
            }
            set {
                this.middleNameField = value;
            }
        }
        
        /// <remarks/>
        public string lastName {
            get {
                return this.lastNameField;
            }
            set {
                this.lastNameField = value;
            }
        }
        
        /// <remarks/>
        public string prefix {
            get {
                return this.prefixField;
            }
            set {
                this.prefixField = value;
            }
        }
        
        /// <remarks/>
        public string suffix {
            get {
                return this.suffixField;
            }
            set {
                this.suffixField = value;
            }
        }
        
        /// <remarks/>
        public string formattedName {
            get {
                return this.formattedNameField;
            }
            set {
                this.formattedNameField = value;
            }
        }
        
        /// <remarks/>
        public string nickname {
            get {
                return this.nicknameField;
            }
            set {
                this.nicknameField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/TransferOccurrence.xsd")]
    public partial class TransferOccurrence : FundsTransferOccurrenceBase {
        
        private Card fromCardField;
        
        private Card toCardField;
        
        /// <remarks/>
        public Card fromCard {
            get {
                return this.fromCardField;
            }
            set {
                this.fromCardField = value;
            }
        }
        
        /// <remarks/>
        public Card toCard {
            get {
                return this.toCardField;
            }
            set {
                this.toCardField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Card.xsd")]
    public partial class Card {
        
        private string cardIdField;
        
        private string cardNumberField;
        
        private CardType cardTypeField;
        
        private string cardSubTypeField;
        
        private System.DateTime expirationDateField;
        
        private bool expirationDateFieldSpecified;
        
        private string pinField;
        
        private string cvv2Field;
        
        private LinkedAccount[] linkedAccountListField;
        
        private string partyIdField;
        
        private string overrideAddressContactIdField;
        
        private string[] nameOnCardField;
        
        private System.DateTime activationDateTimeField;
        
        private bool activationDateTimeFieldSpecified;
        
        private CardStatus cardStatusField;
        
        private string blockedReasonField;
        
        private System.DateTime blockedDateTimeField;
        
        private bool blockedDateTimeFieldSpecified;
        
        private string virtualNumberField;
        
        private string[] merchantCountryCodesField;
        
        private ArtifactId cardDesignImageArtifactIdField;
        
        /// <remarks/>
        public string cardId {
            get {
                return this.cardIdField;
            }
            set {
                this.cardIdField = value;
            }
        }
        
        /// <remarks/>
        public string cardNumber {
            get {
                return this.cardNumberField;
            }
            set {
                this.cardNumberField = value;
            }
        }
        
        /// <remarks/>
        public CardType cardType {
            get {
                return this.cardTypeField;
            }
            set {
                this.cardTypeField = value;
            }
        }
        
        /// <remarks/>
        public string cardSubType {
            get {
                return this.cardSubTypeField;
            }
            set {
                this.cardSubTypeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime expirationDate {
            get {
                return this.expirationDateField;
            }
            set {
                this.expirationDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool expirationDateSpecified {
            get {
                return this.expirationDateFieldSpecified;
            }
            set {
                this.expirationDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string pin {
            get {
                return this.pinField;
            }
            set {
                this.pinField = value;
            }
        }
        
        /// <remarks/>
        public string cvv2 {
            get {
                return this.cvv2Field;
            }
            set {
                this.cvv2Field = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("linkedAccount", IsNullable=false)]
        public LinkedAccount[] linkedAccountList {
            get {
                return this.linkedAccountListField;
            }
            set {
                this.linkedAccountListField = value;
            }
        }
        
        /// <remarks/>
        public string partyId {
            get {
                return this.partyIdField;
            }
            set {
                this.partyIdField = value;
            }
        }
        
        /// <remarks/>
        public string overrideAddressContactId {
            get {
                return this.overrideAddressContactIdField;
            }
            set {
                this.overrideAddressContactIdField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("nameOnCard")]
        public string[] nameOnCard {
            get {
                return this.nameOnCardField;
            }
            set {
                this.nameOnCardField = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime activationDateTime {
            get {
                return this.activationDateTimeField;
            }
            set {
                this.activationDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool activationDateTimeSpecified {
            get {
                return this.activationDateTimeFieldSpecified;
            }
            set {
                this.activationDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public CardStatus cardStatus {
            get {
                return this.cardStatusField;
            }
            set {
                this.cardStatusField = value;
            }
        }
        
        /// <remarks/>
        public string blockedReason {
            get {
                return this.blockedReasonField;
            }
            set {
                this.blockedReasonField = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime blockedDateTime {
            get {
                return this.blockedDateTimeField;
            }
            set {
                this.blockedDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool blockedDateTimeSpecified {
            get {
                return this.blockedDateTimeFieldSpecified;
            }
            set {
                this.blockedDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string virtualNumber {
            get {
                return this.virtualNumberField;
            }
            set {
                this.virtualNumberField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("countryCode", IsNullable=false)]
        public string[] merchantCountryCodes {
            get {
                return this.merchantCountryCodesField;
            }
            set {
                this.merchantCountryCodesField = value;
            }
        }
        
        /// <remarks/>
        public ArtifactId cardDesignImageArtifactId {
            get {
                return this.cardDesignImageArtifactIdField;
            }
            set {
                this.cardDesignImageArtifactIdField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Card.xsd")]
    public enum CardType {
        
        /// <remarks/>
        Atm,
        
        /// <remarks/>
        Credit,
        
        /// <remarks/>
        Debit,
        
        /// <remarks/>
        Prepaid,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Card.xsd")]
    public partial class LinkedAccount {
        
        private string accountIdField;
        
        private string priorityField;
        
        /// <remarks/>
        public string accountId {
            get {
                return this.accountIdField;
            }
            set {
                this.accountIdField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="positiveInteger")]
        public string priority {
            get {
                return this.priorityField;
            }
            set {
                this.priorityField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Card.xsd")]
    public enum CardStatus {
        
        /// <remarks/>
        Inactive,
        
        /// <remarks/>
        Active,
        
        /// <remarks/>
        Blocked,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/BillPaymentOccurrenceMessage.xsd")]
    public partial class BillPaymentOccurrenceMessage {
        
        private MessageContext messageContextField;
        
        private BillPaymentFilter billPaymentFilterField;
        
        private BillPaymentOccurrence[] billPaymentListField;
        
        /// <remarks/>
        public MessageContext messageContext {
            get {
                return this.messageContextField;
            }
            set {
                this.messageContextField = value;
            }
        }
        
        /// <remarks/>
        public BillPaymentFilter billPaymentFilter {
            get {
                return this.billPaymentFilterField;
            }
            set {
                this.billPaymentFilterField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("billPaymentOccurrence", Namespace="http://cufxstandards.com/v3/BillPaymentOccurrence.xsd", IsNullable=false)]
        public BillPaymentOccurrence[] billPaymentList {
            get {
                return this.billPaymentListField;
            }
            set {
                this.billPaymentListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/BillPaymentRecurringMessage.xsd")]
    public partial class BillPaymentRecurringMessage {
        
        private MessageContext messageContextField;
        
        private BillPaymentFilter billPaymentFilterField;
        
        private BillPaymentRecurring[] billPaymentRecurringListField;
        
        /// <remarks/>
        public MessageContext messageContext {
            get {
                return this.messageContextField;
            }
            set {
                this.messageContextField = value;
            }
        }
        
        /// <remarks/>
        public BillPaymentFilter billPaymentFilter {
            get {
                return this.billPaymentFilterField;
            }
            set {
                this.billPaymentFilterField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("billPaymentRecurring", Namespace="http://cufxstandards.com/v3/BillPaymentRecurring.xsd", IsNullable=false)]
        public BillPaymentRecurring[] billPaymentRecurringList {
            get {
                return this.billPaymentRecurringListField;
            }
            set {
                this.billPaymentRecurringListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/BillPaymentRecurring.xsd")]
    public partial class BillPaymentRecurring : FundsTransferRecurringBase {
        
        private string billPayeeIdField;
        
        private System.DateTime nextDebitDateTimeField;
        
        private bool nextDebitDateTimeFieldSpecified;
        
        /// <remarks/>
        public string billPayeeId {
            get {
                return this.billPayeeIdField;
            }
            set {
                this.billPayeeIdField = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime nextDebitDateTime {
            get {
                return this.nextDebitDateTimeField;
            }
            set {
                this.nextDebitDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool nextDebitDateTimeSpecified {
            get {
                return this.nextDebitDateTimeFieldSpecified;
            }
            set {
                this.nextDebitDateTimeFieldSpecified = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/FundsTransferRecurringBase.xsd")]
    public abstract partial class FundsTransferRecurringBase {
        
        private string recurringIdField;
        
        private Money recurringAmountField;
        
        private string recurringFromAccountIdField;
        
        private string recurringToAccountIdField;
        
        private RecurringStatus recurringStatusField;
        
        private bool recurringStatusFieldSpecified;
        
        private string recurringMemoField;
        
        private bool isElectronicField;
        
        private bool isElectronicFieldSpecified;
        
        private bool isOpenEndedField;
        
        private bool isOpenEndedFieldSpecified;
        
        private bool isCompletedField;
        
        private bool isCompletedFieldSpecified;
        
        private bool isUserDeletedField;
        
        private bool isUserDeletedFieldSpecified;
        
        private string createdByFiUserIdField;
        
        private IntervalFrequencyType frequencyField;
        
        private bool frequencyFieldSpecified;
        
        private string totalPaymentCountField;
        
        private string paymentsLeftField;
        
        private FundsWithdrawalType fundsWithdrawalTypeField;
        
        private bool fundsWithdrawalTypeFieldSpecified;
        
        private System.DateTime createRequestDateTimeField;
        
        private bool createRequestDateTimeFieldSpecified;
        
        private System.DateTime firstScheduledDateTimeField;
        
        private bool firstScheduledDateTimeFieldSpecified;
        
        private System.DateTime lastModifiedDateField;
        
        private bool lastModifiedDateFieldSpecified;
        
        private System.DateTime completedDateTimeField;
        
        private bool completedDateTimeFieldSpecified;
        
        private string recurringConfirmationCodeField;
        
        private string[] recurringOccurrenceIdListField;
        
        private ValuePair[] customDataField;
        
        /// <remarks/>
        public string recurringId {
            get {
                return this.recurringIdField;
            }
            set {
                this.recurringIdField = value;
            }
        }
        
        /// <remarks/>
        public Money recurringAmount {
            get {
                return this.recurringAmountField;
            }
            set {
                this.recurringAmountField = value;
            }
        }
        
        /// <remarks/>
        public string recurringFromAccountId {
            get {
                return this.recurringFromAccountIdField;
            }
            set {
                this.recurringFromAccountIdField = value;
            }
        }
        
        /// <remarks/>
        public string recurringToAccountId {
            get {
                return this.recurringToAccountIdField;
            }
            set {
                this.recurringToAccountIdField = value;
            }
        }
        
        /// <remarks/>
        public RecurringStatus recurringStatus {
            get {
                return this.recurringStatusField;
            }
            set {
                this.recurringStatusField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool recurringStatusSpecified {
            get {
                return this.recurringStatusFieldSpecified;
            }
            set {
                this.recurringStatusFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string recurringMemo {
            get {
                return this.recurringMemoField;
            }
            set {
                this.recurringMemoField = value;
            }
        }
        
        /// <remarks/>
        public bool isElectronic {
            get {
                return this.isElectronicField;
            }
            set {
                this.isElectronicField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool isElectronicSpecified {
            get {
                return this.isElectronicFieldSpecified;
            }
            set {
                this.isElectronicFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public bool isOpenEnded {
            get {
                return this.isOpenEndedField;
            }
            set {
                this.isOpenEndedField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool isOpenEndedSpecified {
            get {
                return this.isOpenEndedFieldSpecified;
            }
            set {
                this.isOpenEndedFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public bool isCompleted {
            get {
                return this.isCompletedField;
            }
            set {
                this.isCompletedField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool isCompletedSpecified {
            get {
                return this.isCompletedFieldSpecified;
            }
            set {
                this.isCompletedFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public bool isUserDeleted {
            get {
                return this.isUserDeletedField;
            }
            set {
                this.isUserDeletedField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool isUserDeletedSpecified {
            get {
                return this.isUserDeletedFieldSpecified;
            }
            set {
                this.isUserDeletedFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string createdByFiUserId {
            get {
                return this.createdByFiUserIdField;
            }
            set {
                this.createdByFiUserIdField = value;
            }
        }
        
        /// <remarks/>
        public IntervalFrequencyType frequency {
            get {
                return this.frequencyField;
            }
            set {
                this.frequencyField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool frequencySpecified {
            get {
                return this.frequencyFieldSpecified;
            }
            set {
                this.frequencyFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="integer")]
        public string totalPaymentCount {
            get {
                return this.totalPaymentCountField;
            }
            set {
                this.totalPaymentCountField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="integer")]
        public string paymentsLeft {
            get {
                return this.paymentsLeftField;
            }
            set {
                this.paymentsLeftField = value;
            }
        }
        
        /// <remarks/>
        public FundsWithdrawalType fundsWithdrawalType {
            get {
                return this.fundsWithdrawalTypeField;
            }
            set {
                this.fundsWithdrawalTypeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool fundsWithdrawalTypeSpecified {
            get {
                return this.fundsWithdrawalTypeFieldSpecified;
            }
            set {
                this.fundsWithdrawalTypeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime createRequestDateTime {
            get {
                return this.createRequestDateTimeField;
            }
            set {
                this.createRequestDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool createRequestDateTimeSpecified {
            get {
                return this.createRequestDateTimeFieldSpecified;
            }
            set {
                this.createRequestDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime firstScheduledDateTime {
            get {
                return this.firstScheduledDateTimeField;
            }
            set {
                this.firstScheduledDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool firstScheduledDateTimeSpecified {
            get {
                return this.firstScheduledDateTimeFieldSpecified;
            }
            set {
                this.firstScheduledDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime lastModifiedDate {
            get {
                return this.lastModifiedDateField;
            }
            set {
                this.lastModifiedDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool lastModifiedDateSpecified {
            get {
                return this.lastModifiedDateFieldSpecified;
            }
            set {
                this.lastModifiedDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime completedDateTime {
            get {
                return this.completedDateTimeField;
            }
            set {
                this.completedDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool completedDateTimeSpecified {
            get {
                return this.completedDateTimeFieldSpecified;
            }
            set {
                this.completedDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string recurringConfirmationCode {
            get {
                return this.recurringConfirmationCodeField;
            }
            set {
                this.recurringConfirmationCodeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("occurrenceId", Namespace="http://cufxstandards.com/v3/FundsTransferCommonBase.xsd", IsNullable=false)]
        public string[] recurringOccurrenceIdList {
            get {
                return this.recurringOccurrenceIdListField;
            }
            set {
                this.recurringOccurrenceIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("valuePair", Namespace="http://cufxstandards.com/v3/Common.xsd", IsNullable=false)]
        public ValuePair[] customData {
            get {
                return this.customDataField;
            }
            set {
                this.customDataField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/FundsTransferCommonBase.xsd")]
    public enum RecurringStatus {
        
        /// <remarks/>
        Requested,
        
        /// <remarks/>
        Active,
        
        /// <remarks/>
        CancelRequestedByParty,
        
        /// <remarks/>
        CancelRequestedByFinancialInstitution,
        
        /// <remarks/>
        CancelRequestedByPaymentProvider,
        
        /// <remarks/>
        Cancelled,
        
        /// <remarks/>
        Completed,
        
        /// <remarks/>
        Suspended,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Common.xsd")]
    public enum IntervalFrequencyType {
        
        /// <remarks/>
        OneTime,
        
        /// <remarks/>
        PerUse,
        
        /// <remarks/>
        Hourly,
        
        /// <remarks/>
        Daily,
        
        /// <remarks/>
        Weekly,
        
        /// <remarks/>
        Biweekly,
        
        /// <remarks/>
        Monthly,
        
        /// <remarks/>
        SemiMonthly,
        
        /// <remarks/>
        Quarterly,
        
        /// <remarks/>
        SemiAnnually,
        
        /// <remarks/>
        Annually,
        
        /// <remarks/>
        Other,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/TransferRecurring.xsd")]
    public partial class TransferRecurring : FundsTransferRecurringBase {
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Card.xsd")]
    public partial class CardList {
        
        private Card[] cardField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("card")]
        public Card[] card {
            get {
                return this.cardField;
            }
            set {
                this.cardField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/CardFilter.xsd")]
    public partial class CardFilter {
        
        private string[] cardIdListField;
        
        private string[] partyIdListField;
        
        private string[] accountIdListField;
        
        private System.DateTime transactionStartDateTimeField;
        
        private bool transactionStartDateTimeFieldSpecified;
        
        private System.DateTime transactionEndDateTimeField;
        
        private bool transactionEndDateTimeFieldSpecified;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("cardId", Namespace="http://cufxstandards.com/v3/Card.xsd", IsNullable=false)]
        public string[] cardIdList {
            get {
                return this.cardIdListField;
            }
            set {
                this.cardIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("partyId", Namespace="http://cufxstandards.com/v3/Party.xsd", IsNullable=false)]
        public string[] partyIdList {
            get {
                return this.partyIdListField;
            }
            set {
                this.partyIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("accountId", Namespace="http://cufxstandards.com/v3/Account.xsd", IsNullable=false)]
        public string[] accountIdList {
            get {
                return this.accountIdListField;
            }
            set {
                this.accountIdListField = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime transactionStartDateTime {
            get {
                return this.transactionStartDateTimeField;
            }
            set {
                this.transactionStartDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool transactionStartDateTimeSpecified {
            get {
                return this.transactionStartDateTimeFieldSpecified;
            }
            set {
                this.transactionStartDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime transactionEndDateTime {
            get {
                return this.transactionEndDateTimeField;
            }
            set {
                this.transactionEndDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool transactionEndDateTimeSpecified {
            get {
                return this.transactionEndDateTimeFieldSpecified;
            }
            set {
                this.transactionEndDateTimeFieldSpecified = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/CardMessage.xsd")]
    public partial class CardMessage {
        
        private MessageContext messageContextField;
        
        private CardFilter cardFilterField;
        
        private Card[] cardListField;
        
        /// <remarks/>
        public MessageContext messageContext {
            get {
                return this.messageContextField;
            }
            set {
                this.messageContextField = value;
            }
        }
        
        /// <remarks/>
        public CardFilter cardFilter {
            get {
                return this.cardFilterField;
            }
            set {
                this.cardFilterField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("card", Namespace="http://cufxstandards.com/v3/Card.xsd", IsNullable=false)]
        public Card[] cardList {
            get {
                return this.cardListField;
            }
            set {
                this.cardListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Collateral.xsd")]
    public partial class CollateralList {
        
        private Collateral[] collateralField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("collateral")]
        public Collateral[] collateral {
            get {
                return this.collateralField;
            }
            set {
                this.collateralField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/CollateralFilter.xsd")]
    public partial class CollateralFilter {
        
        private string[] collateralIdListField;
        
        private string descriptionField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("collateralId", Namespace="http://cufxstandards.com/v3/Collateral.xsd", IsNullable=false)]
        public string[] collateralIdList {
            get {
                return this.collateralIdListField;
            }
            set {
                this.collateralIdListField = value;
            }
        }
        
        /// <remarks/>
        public string description {
            get {
                return this.descriptionField;
            }
            set {
                this.descriptionField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/CollateralMessage.xsd")]
    public partial class CollateralMessage {
        
        private MessageContext messageContextField;
        
        private CollateralFilter collateralFilterField;
        
        private Collateral[] collateralListField;
        
        /// <remarks/>
        public MessageContext messageContext {
            get {
                return this.messageContextField;
            }
            set {
                this.messageContextField = value;
            }
        }
        
        /// <remarks/>
        public CollateralFilter collateralFilter {
            get {
                return this.collateralFilterField;
            }
            set {
                this.collateralFilterField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("collateral", Namespace="http://cufxstandards.com/v3/Collateral.xsd", IsNullable=false)]
        public Collateral[] collateralList {
            get {
                return this.collateralListField;
            }
            set {
                this.collateralListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Configuration.xsd")]
    public partial class ConfigurationList {
        
        private Configuration[] configurationField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("configuration")]
        public Configuration[] configuration {
            get {
                return this.configurationField;
            }
            set {
                this.configurationField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Configuration.xsd")]
    public partial class Configuration {
        
        private string[] fiIdListField;
        
        private string endPointTimeZoneUTOffsetField;
        
        private Service[] serviceListField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("fiId", IsNullable=false)]
        public string[] fiIdList {
            get {
                return this.fiIdListField;
            }
            set {
                this.fiIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="integer")]
        public string endPointTimeZoneUTOffset {
            get {
                return this.endPointTimeZoneUTOffsetField;
            }
            set {
                this.endPointTimeZoneUTOffsetField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("service", IsNullable=false)]
        public Service[] serviceList {
            get {
                return this.serviceListField;
            }
            set {
                this.serviceListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Configuration.xsd")]
    public partial class Service {
        
        private ServiceName serviceNameField;
        
        private Protocol protocolField;
        
        private CufxVersion cufxVersionField;
        
        private string wsdlUriField;
        
        private ISOCurrencyCodeType[] currencySupportListField;
        
        private string[] acceptLanguageListField;
        
        private FieldNotSupported[] fieldNotSupportedListField;
        
        private string serviceTimeZoneUTOffsetField;
        
        private SystemStatus statusField;
        
        private bool statusFieldSpecified;
        
        private CustomDataUse[] customDataUseListField;
        
        private MethodList methodListField;
        
        /// <remarks/>
        public ServiceName serviceName {
            get {
                return this.serviceNameField;
            }
            set {
                this.serviceNameField = value;
            }
        }
        
        /// <remarks/>
        public Protocol protocol {
            get {
                return this.protocolField;
            }
            set {
                this.protocolField = value;
            }
        }
        
        /// <remarks/>
        public CufxVersion cufxVersion {
            get {
                return this.cufxVersionField;
            }
            set {
                this.cufxVersionField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="anyURI")]
        public string wsdlUri {
            get {
                return this.wsdlUriField;
            }
            set {
                this.wsdlUriField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("currencySupport", IsNullable=false)]
        public ISOCurrencyCodeType[] currencySupportList {
            get {
                return this.currencySupportListField;
            }
            set {
                this.currencySupportListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("acceptLanguage", IsNullable=false)]
        public string[] acceptLanguageList {
            get {
                return this.acceptLanguageListField;
            }
            set {
                this.acceptLanguageListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("fieldNotSupported", IsNullable=false)]
        public FieldNotSupported[] fieldNotSupportedList {
            get {
                return this.fieldNotSupportedListField;
            }
            set {
                this.fieldNotSupportedListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="integer")]
        public string serviceTimeZoneUTOffset {
            get {
                return this.serviceTimeZoneUTOffsetField;
            }
            set {
                this.serviceTimeZoneUTOffsetField = value;
            }
        }
        
        /// <remarks/>
        public SystemStatus status {
            get {
                return this.statusField;
            }
            set {
                this.statusField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool statusSpecified {
            get {
                return this.statusFieldSpecified;
            }
            set {
                this.statusFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("customDataUse", IsNullable=false)]
        public CustomDataUse[] customDataUseList {
            get {
                return this.customDataUseListField;
            }
            set {
                this.customDataUseListField = value;
            }
        }
        
        /// <remarks/>
        public MethodList methodList {
            get {
                return this.methodListField;
            }
            set {
                this.methodListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Configuration.xsd")]
    public enum ServiceName {
        
        /// <remarks/>
        Account,
        
        /// <remarks/>
        Application,
        
        /// <remarks/>
        Artifact,
        
        /// <remarks/>
        Card,
        
        /// <remarks/>
        Configuration,
        
        /// <remarks/>
        Contact,
        
        /// <remarks/>
        CredentialGroup,
        
        /// <remarks/>
        Deposit,
        
        /// <remarks/>
        DepositFunding,
        
        /// <remarks/>
        EligibilityRequirement,
        
        /// <remarks/>
        Loan,
        
        /// <remarks/>
        LoanDisbursement,
        
        /// <remarks/>
        OverdraftPriority,
        
        /// <remarks/>
        Party,
        
        /// <remarks/>
        PartyAssociation,
        
        /// <remarks/>
        Preference,
        
        /// <remarks/>
        ProductOffering,
        
        /// <remarks/>
        ProductServiceRequest,
        
        /// <remarks/>
        Relationship,
        
        /// <remarks/>
        SimpleValidationRequest,
        
        /// <remarks/>
        CreditReporting,
        
        /// <remarks/>
        Identification,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Configuration.xsd")]
    public enum Protocol {
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("REST-XML")]
        RESTXML,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("REST-JSON")]
        RESTJSON,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("SOAP1.1")]
        SOAP11,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("SOAP1.2")]
        SOAP12,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Configuration.xsd")]
    public enum CufxVersion {
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("1.0.3")]
        Item103,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("1.0.6")]
        Item106,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("2.0.0")]
        Item200,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Configuration.xsd")]
    public partial class FieldNotSupported {
        
        private string xsdFileField;
        
        private string xsdXPathField;
        
        private string commentField;
        
        /// <remarks/>
        public string xsdFile {
            get {
                return this.xsdFileField;
            }
            set {
                this.xsdFileField = value;
            }
        }
        
        /// <remarks/>
        public string xsdXPath {
            get {
                return this.xsdXPathField;
            }
            set {
                this.xsdXPathField = value;
            }
        }
        
        /// <remarks/>
        public string comment {
            get {
                return this.commentField;
            }
            set {
                this.commentField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Configuration.xsd")]
    public enum SystemStatus {
        
        /// <remarks/>
        Online,
        
        /// <remarks/>
        OffLine,
        
        /// <remarks/>
        MemoPost,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Configuration.xsd")]
    public partial class CustomDataUse {
        
        private string xsdFileField;
        
        private string xsdXPathField;
        
        private string[] nameField;
        
        private string commentField;
        
        /// <remarks/>
        public string xsdFile {
            get {
                return this.xsdFileField;
            }
            set {
                this.xsdFileField = value;
            }
        }
        
        /// <remarks/>
        public string xsdXPath {
            get {
                return this.xsdXPathField;
            }
            set {
                this.xsdXPathField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("name")]
        public string[] name {
            get {
                return this.nameField;
            }
            set {
                this.nameField = value;
            }
        }
        
        /// <remarks/>
        public string comment {
            get {
                return this.commentField;
            }
            set {
                this.commentField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Configuration.xsd")]
    public partial class MethodList {
        
        private Method methodField;
        
        /// <remarks/>
        public Method method {
            get {
                return this.methodField;
            }
            set {
                this.methodField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Configuration.xsd")]
    public partial class Method {
        
        private MethodName methodNameField;
        
        private string uriField;
        
        private MethodDependencyList methodDependencyListField;
        
        /// <remarks/>
        public MethodName methodName {
            get {
                return this.methodNameField;
            }
            set {
                this.methodNameField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="anyURI")]
        public string uri {
            get {
                return this.uriField;
            }
            set {
                this.uriField = value;
            }
        }
        
        /// <remarks/>
        public MethodDependencyList methodDependencyList {
            get {
                return this.methodDependencyListField;
            }
            set {
                this.methodDependencyListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Configuration.xsd")]
    public enum MethodName {
        
        /// <remarks/>
        Create,
        
        /// <remarks/>
        Read,
        
        /// <remarks/>
        Update,
        
        /// <remarks/>
        Delete,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Configuration.xsd")]
    public partial class MethodDependencyList {
        
        private MethodDependency methodDependencyField;
        
        /// <remarks/>
        public MethodDependency methodDependency {
            get {
                return this.methodDependencyField;
            }
            set {
                this.methodDependencyField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Configuration.xsd")]
    public partial class MethodDependency {
        
        private ServiceName serviceNameField;
        
        private string methodNameField;
        
        private CufxVersion cufxVersionField;
        
        /// <remarks/>
        public ServiceName serviceName {
            get {
                return this.serviceNameField;
            }
            set {
                this.serviceNameField = value;
            }
        }
        
        /// <remarks/>
        public string methodName {
            get {
                return this.methodNameField;
            }
            set {
                this.methodNameField = value;
            }
        }
        
        /// <remarks/>
        public CufxVersion cufxVersion {
            get {
                return this.cufxVersionField;
            }
            set {
                this.cufxVersionField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Contact.xsd")]
    public partial class ContactList {
        
        private Contact[] contactField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("contact")]
        public Contact[] contact {
            get {
                return this.contactField;
            }
            set {
                this.contactField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/ContactFilter.xsd")]
    public partial class ContactFilter {
        
        private string[] contactIdListField;
        
        private string[] partyIdListField;
        
        private string[] relationshipIdListField;
        
        private string[] accountIdListField;
        
        private string[] taxIdListField;
        
        private ContactType[] contactTypeListField;
        
        private bool badContactPointField;
        
        private bool badContactPointFieldSpecified;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("contactId", Namespace="http://cufxstandards.com/v3/Contact.xsd", IsNullable=false)]
        public string[] contactIdList {
            get {
                return this.contactIdListField;
            }
            set {
                this.contactIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("partyId", Namespace="http://cufxstandards.com/v3/Party.xsd", IsNullable=false)]
        public string[] partyIdList {
            get {
                return this.partyIdListField;
            }
            set {
                this.partyIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("relationshipId", Namespace="http://cufxstandards.com/v3/Relationship.xsd", IsNullable=false)]
        public string[] relationshipIdList {
            get {
                return this.relationshipIdListField;
            }
            set {
                this.relationshipIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("accountId", Namespace="http://cufxstandards.com/v3/Account.xsd", IsNullable=false)]
        public string[] accountIdList {
            get {
                return this.accountIdListField;
            }
            set {
                this.accountIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("taxId", Namespace="http://cufxstandards.com/v3/Party.xsd", DataType="token", IsNullable=false)]
        public string[] taxIdList {
            get {
                return this.taxIdListField;
            }
            set {
                this.taxIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("contactType", Namespace="http://cufxstandards.com/v3/Contact.xsd", IsNullable=false)]
        public ContactType[] contactTypeList {
            get {
                return this.contactTypeListField;
            }
            set {
                this.contactTypeListField = value;
            }
        }
        
        /// <remarks/>
        public bool badContactPoint {
            get {
                return this.badContactPointField;
            }
            set {
                this.badContactPointField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool badContactPointSpecified {
            get {
                return this.badContactPointFieldSpecified;
            }
            set {
                this.badContactPointFieldSpecified = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/ContactMessage.xsd")]
    public partial class ContactMessage {
        
        private MessageContext messageContextField;
        
        private ContactFilter contactFilterField;
        
        private Contact[] contactListField;
        
        /// <remarks/>
        public MessageContext messageContext {
            get {
                return this.messageContextField;
            }
            set {
                this.messageContextField = value;
            }
        }
        
        /// <remarks/>
        public ContactFilter contactFilter {
            get {
                return this.contactFilterField;
            }
            set {
                this.contactFilterField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("contact", Namespace="http://cufxstandards.com/v3/Contact.xsd", IsNullable=false)]
        public Contact[] contactList {
            get {
                return this.contactListField;
            }
            set {
                this.contactListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/CredentialGroup.xsd")]
    public partial class CredentialGroupList {
        
        private CredentialGroup[] credentialGroupField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("credentialGroup", IsNullable=true)]
        public CredentialGroup[] credentialGroup {
            get {
                return this.credentialGroupField;
            }
            set {
                this.credentialGroupField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/CredentialGroup.xsd")]
    public partial class CredentialGroup {
        
        private string fiUserIdField;
        
        private Credential[] credentialListField;
        
        private DeliveryChannel[] deliveryChannelListField;
        
        private string verifiedCredentialGroupTokenField;
        
        /// <remarks/>
        public string fiUserId {
            get {
                return this.fiUserIdField;
            }
            set {
                this.fiUserIdField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("credential", IsNullable=false)]
        public Credential[] credentialList {
            get {
                return this.credentialListField;
            }
            set {
                this.credentialListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("deliveryChannel", IsNullable=false)]
        public DeliveryChannel[] deliveryChannelList {
            get {
                return this.deliveryChannelListField;
            }
            set {
                this.deliveryChannelListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="NMTOKEN")]
        public string verifiedCredentialGroupToken {
            get {
                return this.verifiedCredentialGroupTokenField;
            }
            set {
                this.verifiedCredentialGroupTokenField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/CredentialGroup.xsd")]
    public partial class Credential {
        
        private string credentialIdField;
        
        private string relatedCredentialIdField;
        
        private CredentialType credentialTypeField;
        
        private bool encryptedField;
        
        private bool encryptedFieldSpecified;
        
        private string valueField;
        
        private System.DateTime expirationDateTimeField;
        
        private bool expirationDateTimeFieldSpecified;
        
        private bool temporaryField;
        
        private bool temporaryFieldSpecified;
        
        private ValuePair[] customDataField;
        
        /// <remarks/>
        public string credentialId {
            get {
                return this.credentialIdField;
            }
            set {
                this.credentialIdField = value;
            }
        }
        
        /// <remarks/>
        public string relatedCredentialId {
            get {
                return this.relatedCredentialIdField;
            }
            set {
                this.relatedCredentialIdField = value;
            }
        }
        
        /// <remarks/>
        public CredentialType credentialType {
            get {
                return this.credentialTypeField;
            }
            set {
                this.credentialTypeField = value;
            }
        }
        
        /// <remarks/>
        public bool encrypted {
            get {
                return this.encryptedField;
            }
            set {
                this.encryptedField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool encryptedSpecified {
            get {
                return this.encryptedFieldSpecified;
            }
            set {
                this.encryptedFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="NMTOKEN")]
        public string value {
            get {
                return this.valueField;
            }
            set {
                this.valueField = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime expirationDateTime {
            get {
                return this.expirationDateTimeField;
            }
            set {
                this.expirationDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool expirationDateTimeSpecified {
            get {
                return this.expirationDateTimeFieldSpecified;
            }
            set {
                this.expirationDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public bool temporary {
            get {
                return this.temporaryField;
            }
            set {
                this.temporaryField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool temporarySpecified {
            get {
                return this.temporaryFieldSpecified;
            }
            set {
                this.temporaryFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("valuePair", Namespace="http://cufxstandards.com/v3/Common.xsd", IsNullable=false)]
        public ValuePair[] customData {
            get {
                return this.customDataField;
            }
            set {
                this.customDataField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/CredentialGroup.xsd")]
    public enum CredentialType {
        
        /// <remarks/>
        Username,
        
        /// <remarks/>
        Password,
        
        /// <remarks/>
        AtmPin,
        
        /// <remarks/>
        Codeword,
        
        /// <remarks/>
        Token,
        
        /// <remarks/>
        SecurityQuestion,
        
        /// <remarks/>
        SecurityAnswer,
        
        /// <remarks/>
        ChallengeQuestion,
        
        /// <remarks/>
        ChallengeAnswer,
        
        /// <remarks/>
        AntiphishingPhrase,
        
        /// <remarks/>
        AntiphishingImageUrl,
        
        /// <remarks/>
        AntiphishingImageArtifactId,
        
        /// <remarks/>
        AuthorizationCode,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/CredentialGroup.xsd")]
    public enum DeliveryChannel {
        
        /// <remarks/>
        LiveSupport,
        
        /// <remarks/>
        OnlineBanking,
        
        /// <remarks/>
        Mobile,
        
        /// <remarks/>
        Kiosk,
        
        /// <remarks/>
        ATM,
        
        /// <remarks/>
        IVR,
        
        /// <remarks/>
        Other,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/CredentialGroupFilter.xsd")]
    public partial class CredentialGroupFilter {
        
        private string[] fiUserIdListField;
        
        private string[] partyIdListField;
        
        private string[] relationshipIdListField;
        
        private string[] accountIdListField;
        
        private CredentialType[] credentialTypeListField;
        
        private DeliveryChannel[] deliveryChannelListField;
        
        private CredentialGroup[] verifyCredentialGroupListField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("fiUserId", Namespace="http://cufxstandards.com/v3/CredentialGroup.xsd", IsNullable=false)]
        public string[] fiUserIdList {
            get {
                return this.fiUserIdListField;
            }
            set {
                this.fiUserIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("partyId", Namespace="http://cufxstandards.com/v3/Party.xsd", IsNullable=false)]
        public string[] partyIdList {
            get {
                return this.partyIdListField;
            }
            set {
                this.partyIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("relationshipId", Namespace="http://cufxstandards.com/v3/Relationship.xsd", IsNullable=false)]
        public string[] relationshipIdList {
            get {
                return this.relationshipIdListField;
            }
            set {
                this.relationshipIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("accountId", Namespace="http://cufxstandards.com/v3/Account.xsd", IsNullable=false)]
        public string[] accountIdList {
            get {
                return this.accountIdListField;
            }
            set {
                this.accountIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("credentialType", Namespace="http://cufxstandards.com/v3/CredentialGroup.xsd", IsNullable=false)]
        public CredentialType[] credentialTypeList {
            get {
                return this.credentialTypeListField;
            }
            set {
                this.credentialTypeListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("deliveryChannel", Namespace="http://cufxstandards.com/v3/CredentialGroup.xsd", IsNullable=false)]
        public DeliveryChannel[] deliveryChannelList {
            get {
                return this.deliveryChannelListField;
            }
            set {
                this.deliveryChannelListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("credentialGroup", Namespace="http://cufxstandards.com/v3/CredentialGroup.xsd")]
        public CredentialGroup[] verifyCredentialGroupList {
            get {
                return this.verifyCredentialGroupListField;
            }
            set {
                this.verifyCredentialGroupListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/CredentialGroupMessage.xsd")]
    public partial class CredentialGroupMessage {
        
        private MessageContext messageContextField;
        
        private CredentialGroupFilter credentialGroupFilterField;
        
        private CredentialGroup[] credentialGroupListField;
        
        /// <remarks/>
        public MessageContext messageContext {
            get {
                return this.messageContextField;
            }
            set {
                this.messageContextField = value;
            }
        }
        
        /// <remarks/>
        public CredentialGroupFilter credentialGroupFilter {
            get {
                return this.credentialGroupFilterField;
            }
            set {
                this.credentialGroupFilterField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("credentialGroup", Namespace="http://cufxstandards.com/v3/CredentialGroup.xsd")]
        public CredentialGroup[] credentialGroupList {
            get {
                return this.credentialGroupListField;
            }
            set {
                this.credentialGroupListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/CreditReport.xsd")]
    public partial class CreditReportList {
        
        private CreditReport[] creditReportField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("creditReport")]
        public CreditReport[] creditReport {
            get {
                return this.creditReportField;
            }
            set {
                this.creditReportField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/CreditReportFilter.xsd")]
    public partial class CreditReportFilter {
        
        private string[] creditReportIdListField;
        
        private System.DateTime creditReportStartDateField;
        
        private bool creditReportStartDateFieldSpecified;
        
        private System.DateTime creditReportEndDateField;
        
        private bool creditReportEndDateFieldSpecified;
        
        private string minCreditScoreField;
        
        private string maxCreditScoreField;
        
        private string reportTypeField;
        
        private string partyIdField;
        
        private string taxIdField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("creditReportId", Namespace="http://cufxstandards.com/v3/CreditReport.xsd", IsNullable=false)]
        public string[] creditReportIdList {
            get {
                return this.creditReportIdListField;
            }
            set {
                this.creditReportIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime creditReportStartDate {
            get {
                return this.creditReportStartDateField;
            }
            set {
                this.creditReportStartDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool creditReportStartDateSpecified {
            get {
                return this.creditReportStartDateFieldSpecified;
            }
            set {
                this.creditReportStartDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime creditReportEndDate {
            get {
                return this.creditReportEndDateField;
            }
            set {
                this.creditReportEndDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool creditReportEndDateSpecified {
            get {
                return this.creditReportEndDateFieldSpecified;
            }
            set {
                this.creditReportEndDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="integer")]
        public string minCreditScore {
            get {
                return this.minCreditScoreField;
            }
            set {
                this.minCreditScoreField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="integer")]
        public string maxCreditScore {
            get {
                return this.maxCreditScoreField;
            }
            set {
                this.maxCreditScoreField = value;
            }
        }
        
        /// <remarks/>
        public string reportType {
            get {
                return this.reportTypeField;
            }
            set {
                this.reportTypeField = value;
            }
        }
        
        /// <remarks/>
        public string partyId {
            get {
                return this.partyIdField;
            }
            set {
                this.partyIdField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="token")]
        public string taxId {
            get {
                return this.taxIdField;
            }
            set {
                this.taxIdField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/CreditReportMessage.xsd")]
    public partial class CreditReportMessage {
        
        private MessageContext messageContextField;
        
        private CreditReportFilter creditReportFilterField;
        
        private CreditReport[] creditReportListField;
        
        private CreditReportRequest creditReportRequestField;
        
        /// <remarks/>
        public MessageContext messageContext {
            get {
                return this.messageContextField;
            }
            set {
                this.messageContextField = value;
            }
        }
        
        /// <remarks/>
        public CreditReportFilter creditReportFilter {
            get {
                return this.creditReportFilterField;
            }
            set {
                this.creditReportFilterField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("creditReport", Namespace="http://cufxstandards.com/v3/CreditReport.xsd", IsNullable=false)]
        public CreditReport[] creditReportList {
            get {
                return this.creditReportListField;
            }
            set {
                this.creditReportListField = value;
            }
        }
        
        /// <remarks/>
        public CreditReportRequest creditReportRequest {
            get {
                return this.creditReportRequestField;
            }
            set {
                this.creditReportRequestField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/CreditReportRequest.xsd")]
    public partial class CreditReportRequest {
        
        private Party[] applicantListField;
        
        private string maxReportAgeInDaysField;
        
        private string[] reportTypeListField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("applicant", IsNullable=false)]
        public Party[] applicantList {
            get {
                return this.applicantListField;
            }
            set {
                this.applicantListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="integer")]
        public string maxReportAgeInDays {
            get {
                return this.maxReportAgeInDaysField;
            }
            set {
                this.maxReportAgeInDaysField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("reportType", IsNullable=false)]
        public string[] reportTypeList {
            get {
                return this.reportTypeListField;
            }
            set {
                this.reportTypeListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Deposit.xsd")]
    public partial class DepositList {
        
        private Deposit[] depositField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("deposit")]
        public Deposit[] deposit {
            get {
                return this.depositField;
            }
            set {
                this.depositField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/DepositFunding.xsd")]
    public partial class DepositFundingList {
        
        private DepositFunding[] depositFundingField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("depositFunding")]
        public DepositFunding[] depositFunding {
            get {
                return this.depositFundingField;
            }
            set {
                this.depositFundingField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/DepositFunding.xsd")]
    public partial class DepositFunding {
        
        private string targetAccountIdField;
        
        private string fundingSourceIdField;
        
        private SourceOfFunds sourceOfFundsField;
        
        private bool sourceOfFundsFieldSpecified;
        
        private Money fundingAmountField;
        
        private Money holdAmountField;
        
        private System.DateTime holdExpirationDateField;
        
        private bool holdExpirationDateFieldSpecified;
        
        /// <remarks/>
        public string targetAccountId {
            get {
                return this.targetAccountIdField;
            }
            set {
                this.targetAccountIdField = value;
            }
        }
        
        /// <remarks/>
        public string fundingSourceId {
            get {
                return this.fundingSourceIdField;
            }
            set {
                this.fundingSourceIdField = value;
            }
        }
        
        /// <remarks/>
        public SourceOfFunds sourceOfFunds {
            get {
                return this.sourceOfFundsField;
            }
            set {
                this.sourceOfFundsField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool sourceOfFundsSpecified {
            get {
                return this.sourceOfFundsFieldSpecified;
            }
            set {
                this.sourceOfFundsFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public Money fundingAmount {
            get {
                return this.fundingAmountField;
            }
            set {
                this.fundingAmountField = value;
            }
        }
        
        /// <remarks/>
        public Money holdAmount {
            get {
                return this.holdAmountField;
            }
            set {
                this.holdAmountField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime holdExpirationDate {
            get {
                return this.holdExpirationDateField;
            }
            set {
                this.holdExpirationDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool holdExpirationDateSpecified {
            get {
                return this.holdExpirationDateFieldSpecified;
            }
            set {
                this.holdExpirationDateFieldSpecified = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/DepositFunding.xsd")]
    public enum SourceOfFunds {
        
        /// <remarks/>
        Cash,
        
        /// <remarks/>
        Check,
        
        /// <remarks/>
        Transfer,
        
        /// <remarks/>
        ACH,
        
        /// <remarks/>
        None,
        
        /// <remarks/>
        CreditCard,
        
        /// <remarks/>
        DebitCard,
        
        /// <remarks/>
        Wire,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/DepositFundingMessage.xsd")]
    public partial class DepositFundingMessage {
        
        private MessageContext messageContextField;
        
        private DepositFilter depositFilterField;
        
        private DepositFunding[] depositFundingListField;
        
        /// <remarks/>
        public MessageContext messageContext {
            get {
                return this.messageContextField;
            }
            set {
                this.messageContextField = value;
            }
        }
        
        /// <remarks/>
        public DepositFilter depositFilter {
            get {
                return this.depositFilterField;
            }
            set {
                this.depositFilterField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("depositFunding", Namespace="http://cufxstandards.com/v3/DepositFunding.xsd", IsNullable=false)]
        public DepositFunding[] depositFundingList {
            get {
                return this.depositFundingListField;
            }
            set {
                this.depositFundingListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/DepositMessage.xsd")]
    public partial class DepositMessage {
        
        private MessageContext messageContextField;
        
        private DepositFilter depositFilterField;
        
        private Deposit[] depositListField;
        
        /// <remarks/>
        public MessageContext messageContext {
            get {
                return this.messageContextField;
            }
            set {
                this.messageContextField = value;
            }
        }
        
        /// <remarks/>
        public DepositFilter depositFilter {
            get {
                return this.depositFilterField;
            }
            set {
                this.depositFilterField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("deposit", Namespace="http://cufxstandards.com/v3/Deposit.xsd", IsNullable=false)]
        public Deposit[] depositList {
            get {
                return this.depositListField;
            }
            set {
                this.depositListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Document.xsd")]
    public partial class DocumentList {
        
        private Document[] documentField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("document")]
        public Document[] document {
            get {
                return this.documentField;
            }
            set {
                this.documentField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Document.xsd")]
    public partial class Document {
        
        private string documentIdField;
        
        private string documentTitleField;
        
        private DocumentType documentTypeField;
        
        private bool documentTypeFieldSpecified;
        
        private string documentSubTypeField;
        
        private ValuePair[] appliesToField;
        
        private DocumentStatus documentStatusField;
        
        private bool documentStatusFieldSpecified;
        
        private string documentVersionField;
        
        private System.DateTime effectiveDateTimeField;
        
        private bool effectiveDateTimeFieldSpecified;
        
        private System.DateTime expirationDateTimeField;
        
        private bool expirationDateTimeFieldSpecified;
        
        private ArtifactId[] documentArtifactIdListField;
        
        private string[] partyIdListField;
        
        private string relationshipIdField;
        
        private string[] accountIdListField;
        
        private bool prefillableField;
        
        private bool prefillableFieldSpecified;
        
        private bool prefilledField;
        
        private bool prefilledFieldSpecified;
        
        private DocumentSignature[] documentSignatureListField;
        
        private Witness[] witnessListField;
        
        private ValuePair[] customDataField;
        
        /// <remarks/>
        public string documentId {
            get {
                return this.documentIdField;
            }
            set {
                this.documentIdField = value;
            }
        }
        
        /// <remarks/>
        public string documentTitle {
            get {
                return this.documentTitleField;
            }
            set {
                this.documentTitleField = value;
            }
        }
        
        /// <remarks/>
        public DocumentType documentType {
            get {
                return this.documentTypeField;
            }
            set {
                this.documentTypeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool documentTypeSpecified {
            get {
                return this.documentTypeFieldSpecified;
            }
            set {
                this.documentTypeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string documentSubType {
            get {
                return this.documentSubTypeField;
            }
            set {
                this.documentSubTypeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("valuePair", Namespace="http://cufxstandards.com/v3/Common.xsd", IsNullable=false)]
        public ValuePair[] appliesTo {
            get {
                return this.appliesToField;
            }
            set {
                this.appliesToField = value;
            }
        }
        
        /// <remarks/>
        public DocumentStatus documentStatus {
            get {
                return this.documentStatusField;
            }
            set {
                this.documentStatusField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool documentStatusSpecified {
            get {
                return this.documentStatusFieldSpecified;
            }
            set {
                this.documentStatusFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string documentVersion {
            get {
                return this.documentVersionField;
            }
            set {
                this.documentVersionField = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime effectiveDateTime {
            get {
                return this.effectiveDateTimeField;
            }
            set {
                this.effectiveDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool effectiveDateTimeSpecified {
            get {
                return this.effectiveDateTimeFieldSpecified;
            }
            set {
                this.effectiveDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime expirationDateTime {
            get {
                return this.expirationDateTimeField;
            }
            set {
                this.expirationDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool expirationDateTimeSpecified {
            get {
                return this.expirationDateTimeFieldSpecified;
            }
            set {
                this.expirationDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("artifactId", Namespace="http://cufxstandards.com/v3/Artifact.xsd")]
        public ArtifactId[] documentArtifactIdList {
            get {
                return this.documentArtifactIdListField;
            }
            set {
                this.documentArtifactIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("partyId", Namespace="http://cufxstandards.com/v3/Party.xsd", IsNullable=false)]
        public string[] partyIdList {
            get {
                return this.partyIdListField;
            }
            set {
                this.partyIdListField = value;
            }
        }
        
        /// <remarks/>
        public string relationshipId {
            get {
                return this.relationshipIdField;
            }
            set {
                this.relationshipIdField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("accountId", Namespace="http://cufxstandards.com/v3/Account.xsd", IsNullable=false)]
        public string[] accountIdList {
            get {
                return this.accountIdListField;
            }
            set {
                this.accountIdListField = value;
            }
        }
        
        /// <remarks/>
        public bool prefillable {
            get {
                return this.prefillableField;
            }
            set {
                this.prefillableField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool prefillableSpecified {
            get {
                return this.prefillableFieldSpecified;
            }
            set {
                this.prefillableFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public bool prefilled {
            get {
                return this.prefilledField;
            }
            set {
                this.prefilledField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool prefilledSpecified {
            get {
                return this.prefilledFieldSpecified;
            }
            set {
                this.prefilledFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("documentSignature", IsNullable=false)]
        public DocumentSignature[] documentSignatureList {
            get {
                return this.documentSignatureListField;
            }
            set {
                this.documentSignatureListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("witness", IsNullable=false)]
        public Witness[] witnessList {
            get {
                return this.witnessListField;
            }
            set {
                this.witnessListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("valuePair", Namespace="http://cufxstandards.com/v3/Common.xsd", IsNullable=false)]
        public ValuePair[] customData {
            get {
                return this.customDataField;
            }
            set {
                this.customDataField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Document.xsd")]
    public enum DocumentType {
        
        /// <remarks/>
        CheckImage,
        
        /// <remarks/>
        Disclosure,
        
        /// <remarks/>
        IdentificationDocument,
        
        /// <remarks/>
        LoanDocument,
        
        /// <remarks/>
        MembershipDocument,
        
        /// <remarks/>
        Notice,
        
        /// <remarks/>
        Receipt,
        
        /// <remarks/>
        Statement,
        
        /// <remarks/>
        Report,
        
        /// <remarks/>
        SignatureCard,
        
        /// <remarks/>
        TaxForm,
        
        /// <remarks/>
        Other,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Document.xsd")]
    public enum DocumentStatus {
        
        /// <remarks/>
        Template,
        
        /// <remarks/>
        Sent,
        
        /// <remarks/>
        Viewed,
        
        /// <remarks/>
        Returned,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Document.xsd")]
    public partial class DocumentSignature {
        
        private DocumentSignatureType documentSignatureTypeField;
        
        private bool documentSignatureTypeFieldSpecified;
        
        private string signaturePurposeField;
        
        private ArtifactId signatureArtifactIdField;
        
        private System.DateTime documentSignedDateTimeField;
        
        private bool documentSignedDateTimeFieldSpecified;
        
        private string signaturePartyIdField;
        
        private string signedWithIpAddressField;
        
        private string documentDigitalFingerprintField;
        
        private string documentDigitalAlgorithmField;
        
        private string documentDigitalCertificateField;
        
        private string usersPublicKeyField;
        
        private string viewedConfirmationValueField;
        
        private ViewedConfirmationStatus viewedConfirmationStatusField;
        
        private bool viewedConfirmationStatusFieldSpecified;
        
        /// <remarks/>
        public DocumentSignatureType documentSignatureType {
            get {
                return this.documentSignatureTypeField;
            }
            set {
                this.documentSignatureTypeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool documentSignatureTypeSpecified {
            get {
                return this.documentSignatureTypeFieldSpecified;
            }
            set {
                this.documentSignatureTypeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string signaturePurpose {
            get {
                return this.signaturePurposeField;
            }
            set {
                this.signaturePurposeField = value;
            }
        }
        
        /// <remarks/>
        public ArtifactId signatureArtifactId {
            get {
                return this.signatureArtifactIdField;
            }
            set {
                this.signatureArtifactIdField = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime documentSignedDateTime {
            get {
                return this.documentSignedDateTimeField;
            }
            set {
                this.documentSignedDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool documentSignedDateTimeSpecified {
            get {
                return this.documentSignedDateTimeFieldSpecified;
            }
            set {
                this.documentSignedDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string signaturePartyId {
            get {
                return this.signaturePartyIdField;
            }
            set {
                this.signaturePartyIdField = value;
            }
        }
        
        /// <remarks/>
        public string signedWithIpAddress {
            get {
                return this.signedWithIpAddressField;
            }
            set {
                this.signedWithIpAddressField = value;
            }
        }
        
        /// <remarks/>
        public string documentDigitalFingerprint {
            get {
                return this.documentDigitalFingerprintField;
            }
            set {
                this.documentDigitalFingerprintField = value;
            }
        }
        
        /// <remarks/>
        public string documentDigitalAlgorithm {
            get {
                return this.documentDigitalAlgorithmField;
            }
            set {
                this.documentDigitalAlgorithmField = value;
            }
        }
        
        /// <remarks/>
        public string documentDigitalCertificate {
            get {
                return this.documentDigitalCertificateField;
            }
            set {
                this.documentDigitalCertificateField = value;
            }
        }
        
        /// <remarks/>
        public string usersPublicKey {
            get {
                return this.usersPublicKeyField;
            }
            set {
                this.usersPublicKeyField = value;
            }
        }
        
        /// <remarks/>
        public string viewedConfirmationValue {
            get {
                return this.viewedConfirmationValueField;
            }
            set {
                this.viewedConfirmationValueField = value;
            }
        }
        
        /// <remarks/>
        public ViewedConfirmationStatus viewedConfirmationStatus {
            get {
                return this.viewedConfirmationStatusField;
            }
            set {
                this.viewedConfirmationStatusField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool viewedConfirmationStatusSpecified {
            get {
                return this.viewedConfirmationStatusFieldSpecified;
            }
            set {
                this.viewedConfirmationStatusFieldSpecified = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Document.xsd")]
    public enum DocumentSignatureType {
        
        /// <remarks/>
        Unsigned,
        
        /// <remarks/>
        ElectronicSignature,
        
        /// <remarks/>
        SingleClickAcceptance,
        
        /// <remarks/>
        WetSignature,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Document.xsd")]
    public enum ViewedConfirmationStatus {
        
        /// <remarks/>
        Presented,
        
        /// <remarks/>
        Confirmed,
        
        /// <remarks/>
        Failed,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Document.xsd")]
    public partial class Witness {
        
        private string witnessIdField;
        
        private WitnessIdType witnessIdTypeField;
        
        private bool witnessIdTypeFieldSpecified;
        
        /// <remarks/>
        public string witnessId {
            get {
                return this.witnessIdField;
            }
            set {
                this.witnessIdField = value;
            }
        }
        
        /// <remarks/>
        public WitnessIdType witnessIdType {
            get {
                return this.witnessIdTypeField;
            }
            set {
                this.witnessIdTypeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool witnessIdTypeSpecified {
            get {
                return this.witnessIdTypeFieldSpecified;
            }
            set {
                this.witnessIdTypeFieldSpecified = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Document.xsd")]
    public enum WitnessIdType {
        
        /// <remarks/>
        WitnessId,
        
        /// <remarks/>
        VendorEmployeeId,
        
        /// <remarks/>
        SystemAccountId,
        
        /// <remarks/>
        NotaryId,
        
        /// <remarks/>
        Custom,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/DocumentFilter.xsd")]
    public partial class DocumentFilter {
        
        private string[] documentIdListField;
        
        private string[] documentTitleListField;
        
        private DocumentType[] documentTypeListField;
        
        private string[] documentSubTypeListField;
        
        private ValuePair[] appliesToField;
        
        private System.DateTime filterDateTimeField;
        
        private bool filterDateTimeFieldSpecified;
        
        private string documentVersionField;
        
        private string[] partyIdListField;
        
        private string[] relationshipIdListField;
        
        private string[] accountIdListField;
        
        private DocumentStatus[] documentStatusListField;
        
        private DocumentSignatureType[] documentSignatureTypeListField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("documentId", Namespace="http://cufxstandards.com/v3/Document.xsd", IsNullable=false)]
        public string[] documentIdList {
            get {
                return this.documentIdListField;
            }
            set {
                this.documentIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("documentTitle", Namespace="http://cufxstandards.com/v3/Document.xsd", IsNullable=false)]
        public string[] documentTitleList {
            get {
                return this.documentTitleListField;
            }
            set {
                this.documentTitleListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("documentType", Namespace="http://cufxstandards.com/v3/Document.xsd", IsNullable=false)]
        public DocumentType[] documentTypeList {
            get {
                return this.documentTypeListField;
            }
            set {
                this.documentTypeListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("documentSubType", Namespace="http://cufxstandards.com/v3/Document.xsd", IsNullable=false)]
        public string[] documentSubTypeList {
            get {
                return this.documentSubTypeListField;
            }
            set {
                this.documentSubTypeListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("valuePair", Namespace="http://cufxstandards.com/v3/Common.xsd", IsNullable=false)]
        public ValuePair[] appliesTo {
            get {
                return this.appliesToField;
            }
            set {
                this.appliesToField = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime filterDateTime {
            get {
                return this.filterDateTimeField;
            }
            set {
                this.filterDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool filterDateTimeSpecified {
            get {
                return this.filterDateTimeFieldSpecified;
            }
            set {
                this.filterDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string documentVersion {
            get {
                return this.documentVersionField;
            }
            set {
                this.documentVersionField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("partyId", Namespace="http://cufxstandards.com/v3/Party.xsd", IsNullable=false)]
        public string[] partyIdList {
            get {
                return this.partyIdListField;
            }
            set {
                this.partyIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("relationshipId", Namespace="http://cufxstandards.com/v3/Relationship.xsd", IsNullable=false)]
        public string[] relationshipIdList {
            get {
                return this.relationshipIdListField;
            }
            set {
                this.relationshipIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("accountId", Namespace="http://cufxstandards.com/v3/Account.xsd", IsNullable=false)]
        public string[] accountIdList {
            get {
                return this.accountIdListField;
            }
            set {
                this.accountIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("documentStatus", Namespace="http://cufxstandards.com/v3/Document.xsd", IsNullable=false)]
        public DocumentStatus[] documentStatusList {
            get {
                return this.documentStatusListField;
            }
            set {
                this.documentStatusListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("documentSignatureType", Namespace="http://cufxstandards.com/v3/Document.xsd", IsNullable=false)]
        public DocumentSignatureType[] documentSignatureTypeList {
            get {
                return this.documentSignatureTypeListField;
            }
            set {
                this.documentSignatureTypeListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/DocumentMessage.xsd")]
    public partial class DocumentMessage {
        
        private MessageContext messageContextField;
        
        private DocumentFilter documentFilterField;
        
        private Document[] documentListField;
        
        /// <remarks/>
        public MessageContext messageContext {
            get {
                return this.messageContextField;
            }
            set {
                this.messageContextField = value;
            }
        }
        
        /// <remarks/>
        public DocumentFilter documentFilter {
            get {
                return this.documentFilterField;
            }
            set {
                this.documentFilterField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("document", Namespace="http://cufxstandards.com/v3/Document.xsd", IsNullable=false)]
        public Document[] documentList {
            get {
                return this.documentListField;
            }
            set {
                this.documentListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/EligibilityRequirement.xsd")]
    public partial class EligibilityRequirementList {
        
        private EligibilityRequirement[] eligibilityRequirementField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("eligibilityRequirement", IsNullable=true)]
        public EligibilityRequirement[] eligibilityRequirement {
            get {
                return this.eligibilityRequirementField;
            }
            set {
                this.eligibilityRequirementField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/EligibilityRequirement.xsd")]
    public partial class EligibilityRequirement {
        
        private string requirementIdField;
        
        private string descriptionField;
        
        private string[] affinityBrandListField;
        
        private string nextActionRequiredIdField;
        
        private ValuePair[] customDataField;
        
        /// <remarks/>
        public string requirementId {
            get {
                return this.requirementIdField;
            }
            set {
                this.requirementIdField = value;
            }
        }
        
        /// <remarks/>
        public string description {
            get {
                return this.descriptionField;
            }
            set {
                this.descriptionField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("affinityBrand")]
        public string[] affinityBrandList {
            get {
                return this.affinityBrandListField;
            }
            set {
                this.affinityBrandListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="integer")]
        public string nextActionRequiredId {
            get {
                return this.nextActionRequiredIdField;
            }
            set {
                this.nextActionRequiredIdField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("valuePair", Namespace="http://cufxstandards.com/v3/Common.xsd", IsNullable=false)]
        public ValuePair[] customData {
            get {
                return this.customDataField;
            }
            set {
                this.customDataField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/EligibilityRequirementFilter.xsd")]
    public partial class EligibilityRequirementFilter {
        
        private string[] eligibilityRequirementIdListField;
        
        private string[] affinityBrandListField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("requirementId", Namespace="http://cufxstandards.com/v3/EligibilityRequirement.xsd", IsNullable=false)]
        public string[] eligibilityRequirementIdList {
            get {
                return this.eligibilityRequirementIdListField;
            }
            set {
                this.eligibilityRequirementIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("affinityBrand", Namespace="http://cufxstandards.com/v3/EligibilityRequirement.xsd")]
        public string[] affinityBrandList {
            get {
                return this.affinityBrandListField;
            }
            set {
                this.affinityBrandListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/EligibilityRequirementMessage.xsd")]
    public partial class EligibilityRequirementMessage {
        
        private MessageContext messageContextField;
        
        private EligibilityRequirementFilter eligibilityRequirementFilterField;
        
        private EligibilityRequirement[] eligibilityRequirementListField;
        
        /// <remarks/>
        public MessageContext messageContext {
            get {
                return this.messageContextField;
            }
            set {
                this.messageContextField = value;
            }
        }
        
        /// <remarks/>
        public EligibilityRequirementFilter eligibilityRequirementFilter {
            get {
                return this.eligibilityRequirementFilterField;
            }
            set {
                this.eligibilityRequirementFilterField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("eligibilityRequirement", Namespace="http://cufxstandards.com/v3/EligibilityRequirement.xsd")]
        public EligibilityRequirement[] eligibilityRequirementList {
            get {
                return this.eligibilityRequirementListField;
            }
            set {
                this.eligibilityRequirementListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Error.xsd")]
    public partial class ErrorList {
        
        private Error[] errorField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("error", IsNullable=true)]
        public Error[] error {
            get {
                return this.errorField;
            }
            set {
                this.errorField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Error.xsd")]
    public partial class Error {
        
        private Code codeField;
        
        private Type typeField;
        
        private string subCodeField;
        
        private string messageField;
        
        private Substitution[] substitutionListField;
        
        /// <remarks/>
        public Code code {
            get {
                return this.codeField;
            }
            set {
                this.codeField = value;
            }
        }
        
        /// <remarks/>
        public Type type {
            get {
                return this.typeField;
            }
            set {
                this.typeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="integer")]
        public string subCode {
            get {
                return this.subCodeField;
            }
            set {
                this.subCodeField = value;
            }
        }
        
        /// <remarks/>
        public string message {
            get {
                return this.messageField;
            }
            set {
                this.messageField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("substitution", IsNullable=false)]
        public Substitution[] substitutionList {
            get {
                return this.substitutionListField;
            }
            set {
                this.substitutionListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Error.xsd")]
    public enum Code {
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("300")]
        Item300,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("302")]
        Item302,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("400")]
        Item400,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("401")]
        Item401,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("402")]
        Item402,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("403")]
        Item403,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("409")]
        Item409,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("411")]
        Item411,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("412")]
        Item412,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("413")]
        Item413,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("415")]
        Item415,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("422")]
        Item422,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("423")]
        Item423,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("424")]
        Item424,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("425")]
        Item425,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("426")]
        Item426,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("429")]
        Item429,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("430")]
        Item430,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("431")]
        Item431,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("432")]
        Item432,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("433")]
        Item433,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("434")]
        Item434,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("435")]
        Item435,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("436")]
        Item436,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("437")]
        Item437,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("438")]
        Item438,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("439")]
        Item439,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("440")]
        Item440,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("441")]
        Item441,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("442")]
        Item442,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("443")]
        Item443,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("444")]
        Item444,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("445")]
        Item445,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("475")]
        Item475,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("476")]
        Item476,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("497")]
        Item497,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("498")]
        Item498,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("499")]
        Item499,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("503")]
        Item503,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Error.xsd")]
    public enum Type {
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("100")]
        Item100,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("200")]
        Item200,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("300")]
        Item300,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("310")]
        Item310,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("320")]
        Item320,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("400")]
        Item400,
        
        /// <remarks/>
        [System.Xml.Serialization.XmlEnumAttribute("500")]
        Item500,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Error.xsd")]
    public partial class Substitution {
        
        private string idField;
        
        private string valueField;
        
        /// <remarks/>
        public string id {
            get {
                return this.idField;
            }
            set {
                this.idField = value;
            }
        }
        
        /// <remarks/>
        public string value {
            get {
                return this.valueField;
            }
            set {
                this.valueField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/FeeSchedule.xsd")]
    public partial class FeeList {
        
        private Fee[] feeField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("fee")]
        public Fee[] fee {
            get {
                return this.feeField;
            }
            set {
                this.feeField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/FeeSchedule.xsd")]
    public partial class Fee {
        
        private string[] feeIdField;
        
        private string typeField;
        
        private RelatedTo relatedToField;
        
        private IntervalFrequencyType frequencyField;
        
        private string minimumFrequencyField;
        
        private string maximumFrequencyField;
        
        private FeePriceList priceListField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("feeId")]
        public string[] feeId {
            get {
                return this.feeIdField;
            }
            set {
                this.feeIdField = value;
            }
        }
        
        /// <remarks/>
        public string type {
            get {
                return this.typeField;
            }
            set {
                this.typeField = value;
            }
        }
        
        /// <remarks/>
        public RelatedTo relatedTo {
            get {
                return this.relatedToField;
            }
            set {
                this.relatedToField = value;
            }
        }
        
        /// <remarks/>
        public IntervalFrequencyType frequency {
            get {
                return this.frequencyField;
            }
            set {
                this.frequencyField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="integer")]
        public string minimumFrequency {
            get {
                return this.minimumFrequencyField;
            }
            set {
                this.minimumFrequencyField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="integer")]
        public string maximumFrequency {
            get {
                return this.maximumFrequencyField;
            }
            set {
                this.maximumFrequencyField = value;
            }
        }
        
        /// <remarks/>
        public FeePriceList priceList {
            get {
                return this.priceListField;
            }
            set {
                this.priceListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/FeeSchedule.xsd")]
    public partial class RelatedTo {
        
        private string itemField;
        
        private ItemChoiceType1 itemElementNameField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("accountId", typeof(string))]
        [System.Xml.Serialization.XmlElementAttribute("partyId", typeof(string))]
        [System.Xml.Serialization.XmlElementAttribute("relationshipId", typeof(string))]
        [System.Xml.Serialization.XmlChoiceIdentifierAttribute("ItemElementName")]
        public string Item {
            get {
                return this.itemField;
            }
            set {
                this.itemField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public ItemChoiceType1 ItemElementName {
            get {
                return this.itemElementNameField;
            }
            set {
                this.itemElementNameField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/FeeSchedule.xsd", IncludeInSchema=false)]
    public enum ItemChoiceType1 {
        
        /// <remarks/>
        accountId,
        
        /// <remarks/>
        partyId,
        
        /// <remarks/>
        relationshipId,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(AnonymousType=true, Namespace="http://cufxstandards.com/v3/FeeSchedule.xsd")]
    public partial class FeePriceList {
        
        private string descriptionField;
        
        private Money priceField;
        
        /// <remarks/>
        public string description {
            get {
                return this.descriptionField;
            }
            set {
                this.descriptionField = value;
            }
        }
        
        /// <remarks/>
        public Money price {
            get {
                return this.priceField;
            }
            set {
                this.priceField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/FeeScheduleFilter.xsd")]
    public partial class FeeScheduleFilter {
        
        private string[] partyIdListField;
        
        private string[] relationshipIdListField;
        
        private string[] contactIdListField;
        
        private string[] accountIdListField;
        
        private string[] taxIdListField;
        
        private PartyType[] partyTypeListField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("partyId", Namespace="http://cufxstandards.com/v3/Party.xsd", IsNullable=false)]
        public string[] partyIdList {
            get {
                return this.partyIdListField;
            }
            set {
                this.partyIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("relationshipId", Namespace="http://cufxstandards.com/v3/Relationship.xsd", IsNullable=false)]
        public string[] relationshipIdList {
            get {
                return this.relationshipIdListField;
            }
            set {
                this.relationshipIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("contactId", Namespace="http://cufxstandards.com/v3/Contact.xsd", IsNullable=false)]
        public string[] contactIdList {
            get {
                return this.contactIdListField;
            }
            set {
                this.contactIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("accountId", Namespace="http://cufxstandards.com/v3/Account.xsd", IsNullable=false)]
        public string[] accountIdList {
            get {
                return this.accountIdListField;
            }
            set {
                this.accountIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("taxId", Namespace="http://cufxstandards.com/v3/Party.xsd", DataType="token", IsNullable=false)]
        public string[] taxIdList {
            get {
                return this.taxIdListField;
            }
            set {
                this.taxIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("partyType", Namespace="http://cufxstandards.com/v3/Party.xsd", IsNullable=false)]
        public PartyType[] partyTypeList {
            get {
                return this.partyTypeListField;
            }
            set {
                this.partyTypeListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/FeeScheduleMessage.xsd")]
    public partial class FeeScheduleMessage {
        
        private MessageContext messageContextField;
        
        private FeeScheduleFilter feeScheduleFilterField;
        
        private Fee[] feeListField;
        
        /// <remarks/>
        public MessageContext messageContext {
            get {
                return this.messageContextField;
            }
            set {
                this.messageContextField = value;
            }
        }
        
        /// <remarks/>
        public FeeScheduleFilter feeScheduleFilter {
            get {
                return this.feeScheduleFilterField;
            }
            set {
                this.feeScheduleFilterField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("fee", Namespace="http://cufxstandards.com/v3/FeeSchedule.xsd", IsNullable=false)]
        public Fee[] feeList {
            get {
                return this.feeListField;
            }
            set {
                this.feeListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Hold.xsd")]
    public partial class HoldList {
        
        private Hold[] holdField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("hold")]
        public Hold[] hold {
            get {
                return this.holdField;
            }
            set {
                this.holdField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Hold.xsd")]
    public partial class Hold {
        
        private string holdIdField;
        
        private string accountIdField;
        
        private string transactionIdField;
        
        private HoldType holdTypeField;
        
        private bool holdTypeFieldSpecified;
        
        private System.DateTime effectiveDateField;
        
        private bool effectiveDateFieldSpecified;
        
        private System.DateTime expirationDateField;
        
        private bool expirationDateFieldSpecified;
        
        private System.DateTime actualReleaseDateField;
        
        private bool actualReleaseDateFieldSpecified;
        
        private HoldStatus holdStatusField;
        
        private bool holdStatusFieldSpecified;
        
        private string descriptionField;
        
        private Money amountField;
        
        private string payeeField;
        
        private HoldReasonType reasonField;
        
        private bool reasonFieldSpecified;
        
        private string feeIdField;
        
        /// <remarks/>
        public string holdId {
            get {
                return this.holdIdField;
            }
            set {
                this.holdIdField = value;
            }
        }
        
        /// <remarks/>
        public string accountId {
            get {
                return this.accountIdField;
            }
            set {
                this.accountIdField = value;
            }
        }
        
        /// <remarks/>
        public string transactionId {
            get {
                return this.transactionIdField;
            }
            set {
                this.transactionIdField = value;
            }
        }
        
        /// <remarks/>
        public HoldType holdType {
            get {
                return this.holdTypeField;
            }
            set {
                this.holdTypeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool holdTypeSpecified {
            get {
                return this.holdTypeFieldSpecified;
            }
            set {
                this.holdTypeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime effectiveDate {
            get {
                return this.effectiveDateField;
            }
            set {
                this.effectiveDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool effectiveDateSpecified {
            get {
                return this.effectiveDateFieldSpecified;
            }
            set {
                this.effectiveDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime expirationDate {
            get {
                return this.expirationDateField;
            }
            set {
                this.expirationDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool expirationDateSpecified {
            get {
                return this.expirationDateFieldSpecified;
            }
            set {
                this.expirationDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime actualReleaseDate {
            get {
                return this.actualReleaseDateField;
            }
            set {
                this.actualReleaseDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool actualReleaseDateSpecified {
            get {
                return this.actualReleaseDateFieldSpecified;
            }
            set {
                this.actualReleaseDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public HoldStatus holdStatus {
            get {
                return this.holdStatusField;
            }
            set {
                this.holdStatusField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool holdStatusSpecified {
            get {
                return this.holdStatusFieldSpecified;
            }
            set {
                this.holdStatusFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string description {
            get {
                return this.descriptionField;
            }
            set {
                this.descriptionField = value;
            }
        }
        
        /// <remarks/>
        public Money amount {
            get {
                return this.amountField;
            }
            set {
                this.amountField = value;
            }
        }
        
        /// <remarks/>
        public string payee {
            get {
                return this.payeeField;
            }
            set {
                this.payeeField = value;
            }
        }
        
        /// <remarks/>
        public HoldReasonType reason {
            get {
                return this.reasonField;
            }
            set {
                this.reasonField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool reasonSpecified {
            get {
                return this.reasonFieldSpecified;
            }
            set {
                this.reasonFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string feeId {
            get {
                return this.feeIdField;
            }
            set {
                this.feeIdField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Hold.xsd")]
    public enum HoldType {
        
        /// <remarks/>
        GeneralPurpose,
        
        /// <remarks/>
        CheckHold,
        
        /// <remarks/>
        CertifiedDraft,
        
        /// <remarks/>
        StopDraft,
        
        /// <remarks/>
        SignatureAuth,
        
        /// <remarks/>
        PledgeHold,
        
        /// <remarks/>
        StopACH,
        
        /// <remarks/>
        StopDraftVerbal,
        
        /// <remarks/>
        StopACHVerbal,
        
        /// <remarks/>
        RevokeACH,
        
        /// <remarks/>
        MerchantVerification,
        
        /// <remarks/>
        UncollectedFee,
        
        /// <remarks/>
        holdDraft,
        
        /// <remarks/>
        BillPayment,
        
        /// <remarks/>
        UnauthorizedACHStop,
        
        /// <remarks/>
        ACHOrigination,
        
        /// <remarks/>
        ACHDNE,
        
        /// <remarks/>
        PinAuth,
        
        /// <remarks/>
        BusinessBlockACHDebit,
        
        /// <remarks/>
        WireHold,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Hold.xsd")]
    public enum HoldStatus {
        
        /// <remarks/>
        Active,
        
        /// <remarks/>
        InActive,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Hold.xsd")]
    public enum HoldReasonType {
        
        /// <remarks/>
        Unknown,
        
        /// <remarks/>
        Lost,
        
        /// <remarks/>
        Stolen,
        
        /// <remarks/>
        Destroyed,
        
        /// <remarks/>
        NotEndorsed,
        
        /// <remarks/>
        Certified,
        
        /// <remarks/>
        Disputed,
        
        /// <remarks/>
        ReturnedMerchandise,
        
        /// <remarks/>
        StoppedService,
        
        /// <remarks/>
        Other,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/HoldFilter.xsd")]
    public partial class HoldFilter {
        
        private string[] holdIdListField;
        
        private string[] partyIdListField;
        
        private string[] accountIdListField;
        
        private string[] relationshipIdListField;
        
        private string[] transactionIdListField;
        
        private HoldStatus holdStatusField;
        
        private bool holdStatusFieldSpecified;
        
        private HoldType[] holdTypeListField;
        
        private HoldReasonType[] holdReasonTypeListField;
        
        private Money minAmountField;
        
        private Money maxAmountField;
        
        private System.DateTime holdEffectiveStartDateTimeField;
        
        private bool holdEffectiveStartDateTimeFieldSpecified;
        
        private System.DateTime holdEffectiveEndDateTimeField;
        
        private bool holdEffectiveEndDateTimeFieldSpecified;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("holdId", Namespace="http://cufxstandards.com/v3/Hold.xsd", IsNullable=false)]
        public string[] holdIdList {
            get {
                return this.holdIdListField;
            }
            set {
                this.holdIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("partyId", Namespace="http://cufxstandards.com/v3/Party.xsd", IsNullable=false)]
        public string[] partyIdList {
            get {
                return this.partyIdListField;
            }
            set {
                this.partyIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("accountId", Namespace="http://cufxstandards.com/v3/Account.xsd", IsNullable=false)]
        public string[] accountIdList {
            get {
                return this.accountIdListField;
            }
            set {
                this.accountIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("relationshipId", Namespace="http://cufxstandards.com/v3/Relationship.xsd", IsNullable=false)]
        public string[] relationshipIdList {
            get {
                return this.relationshipIdListField;
            }
            set {
                this.relationshipIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("transactionId", Namespace="http://cufxstandards.com/v3/Transaction.xsd", IsNullable=false)]
        public string[] transactionIdList {
            get {
                return this.transactionIdListField;
            }
            set {
                this.transactionIdListField = value;
            }
        }
        
        /// <remarks/>
        public HoldStatus holdStatus {
            get {
                return this.holdStatusField;
            }
            set {
                this.holdStatusField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool holdStatusSpecified {
            get {
                return this.holdStatusFieldSpecified;
            }
            set {
                this.holdStatusFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("holdType", Namespace="http://cufxstandards.com/v3/Hold.xsd", IsNullable=false)]
        public HoldType[] holdTypeList {
            get {
                return this.holdTypeListField;
            }
            set {
                this.holdTypeListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("holdReasonType", Namespace="http://cufxstandards.com/v3/Hold.xsd", IsNullable=false)]
        public HoldReasonType[] holdReasonTypeList {
            get {
                return this.holdReasonTypeListField;
            }
            set {
                this.holdReasonTypeListField = value;
            }
        }
        
        /// <remarks/>
        public Money minAmount {
            get {
                return this.minAmountField;
            }
            set {
                this.minAmountField = value;
            }
        }
        
        /// <remarks/>
        public Money maxAmount {
            get {
                return this.maxAmountField;
            }
            set {
                this.maxAmountField = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime holdEffectiveStartDateTime {
            get {
                return this.holdEffectiveStartDateTimeField;
            }
            set {
                this.holdEffectiveStartDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool holdEffectiveStartDateTimeSpecified {
            get {
                return this.holdEffectiveStartDateTimeFieldSpecified;
            }
            set {
                this.holdEffectiveStartDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime holdEffectiveEndDateTime {
            get {
                return this.holdEffectiveEndDateTimeField;
            }
            set {
                this.holdEffectiveEndDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool holdEffectiveEndDateTimeSpecified {
            get {
                return this.holdEffectiveEndDateTimeFieldSpecified;
            }
            set {
                this.holdEffectiveEndDateTimeFieldSpecified = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/HoldMessage.xsd")]
    public partial class HoldMessage {
        
        private MessageContext messageContextField;
        
        private HoldFilter holdFilterField;
        
        private Hold[] holdListField;
        
        /// <remarks/>
        public MessageContext messageContext {
            get {
                return this.messageContextField;
            }
            set {
                this.messageContextField = value;
            }
        }
        
        /// <remarks/>
        public HoldFilter holdFilter {
            get {
                return this.holdFilterField;
            }
            set {
                this.holdFilterField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("hold", Namespace="http://cufxstandards.com/v3/Hold.xsd", IsNullable=false)]
        public Hold[] holdList {
            get {
                return this.holdListField;
            }
            set {
                this.holdListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Loan.xsd")]
    public partial class LoanList {
        
        private Loan[] loanField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("loan")]
        public Loan[] loan {
            get {
                return this.loanField;
            }
            set {
                this.loanField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/LoanDisbursement.xsd")]
    public partial class LoanDisbursementList {
        
        private LoanDisbursement[] loanDisbursementField;
        
        private string loanIdField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("loanDisbursement")]
        public LoanDisbursement[] loanDisbursement {
            get {
                return this.loanDisbursementField;
            }
            set {
                this.loanDisbursementField = value;
            }
        }
        
        /// <remarks/>
        public string loanId {
            get {
                return this.loanIdField;
            }
            set {
                this.loanIdField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/LoanDisbursement.xsd")]
    public partial class LoanDisbursement {
        
        private LoanDisbursementBase itemField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("loanDisbursementCheck", typeof(LoanDisbursementCheck))]
        [System.Xml.Serialization.XmlElementAttribute("loanDisbursementDealerAch", typeof(LoanDisbursementDealerAch))]
        [System.Xml.Serialization.XmlElementAttribute("loanDisbursementDepositToAccount", typeof(LoanDisbursementDepositToAccount))]
        [System.Xml.Serialization.XmlElementAttribute("loanDisbursementGL", typeof(LoanDisbursementGL))]
        public LoanDisbursementBase Item {
            get {
                return this.itemField;
            }
            set {
                this.itemField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/LoanDisbursement.xsd")]
    public partial class LoanDisbursementCheck : LoanDisbursementBase {
        
        private string[] payeeLinesField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("payeeLines")]
        public string[] payeeLines {
            get {
                return this.payeeLinesField;
            }
            set {
                this.payeeLinesField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/LoanDisbursement.xsd")]
    public partial class LoanDisbursementBase {
        
        private string loanDisbursementIdField;
        
        private decimal amountField;
        
        private string descriptionField;
        
        private string commentField;
        
        /// <remarks/>
        public string loanDisbursementId {
            get {
                return this.loanDisbursementIdField;
            }
            set {
                this.loanDisbursementIdField = value;
            }
        }
        
        /// <remarks/>
        public decimal amount {
            get {
                return this.amountField;
            }
            set {
                this.amountField = value;
            }
        }
        
        /// <remarks/>
        public string description {
            get {
                return this.descriptionField;
            }
            set {
                this.descriptionField = value;
            }
        }
        
        /// <remarks/>
        public string comment {
            get {
                return this.commentField;
            }
            set {
                this.commentField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/LoanDisbursement.xsd")]
    public partial class LoanDisbursementGL : LoanDisbursementBase {
        
        private string glAccountField;
        
        /// <remarks/>
        public string glAccount {
            get {
                return this.glAccountField;
            }
            set {
                this.glAccountField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/LoanDisbursement.xsd")]
    public partial class LoanDisbursementDealerAch : LoanDisbursementBase {
        
        private string dealerIdField;
        
        private string dealerNameField;
        
        private string dealerRtnField;
        
        private string dealerAccountNumberField;
        
        /// <remarks/>
        public string dealerId {
            get {
                return this.dealerIdField;
            }
            set {
                this.dealerIdField = value;
            }
        }
        
        /// <remarks/>
        public string dealerName {
            get {
                return this.dealerNameField;
            }
            set {
                this.dealerNameField = value;
            }
        }
        
        /// <remarks/>
        public string dealerRtn {
            get {
                return this.dealerRtnField;
            }
            set {
                this.dealerRtnField = value;
            }
        }
        
        /// <remarks/>
        public string dealerAccountNumber {
            get {
                return this.dealerAccountNumberField;
            }
            set {
                this.dealerAccountNumberField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/LoanDisbursement.xsd")]
    public partial class LoanDisbursementDepositToAccount : LoanDisbursementBase {
        
        private string toAccountField;
        
        private AccountType toAccountTypeField;
        
        /// <remarks/>
        public string toAccount {
            get {
                return this.toAccountField;
            }
            set {
                this.toAccountField = value;
            }
        }
        
        /// <remarks/>
        public AccountType toAccountType {
            get {
                return this.toAccountTypeField;
            }
            set {
                this.toAccountTypeField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/LoanDisbursementMessage.xsd")]
    public partial class LoanDisbursementMessage {
        
        private MessageContext messageContextField;
        
        private LoanDisbursementList loanDisbursementListField;
        
        /// <remarks/>
        public MessageContext messageContext {
            get {
                return this.messageContextField;
            }
            set {
                this.messageContextField = value;
            }
        }
        
        /// <remarks/>
        public LoanDisbursementList loanDisbursementList {
            get {
                return this.loanDisbursementListField;
            }
            set {
                this.loanDisbursementListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/LoanMessage.xsd")]
    public partial class LoanMessage {
        
        private MessageContext messageContextField;
        
        private LoanFilter loanFilterField;
        
        private Loan[] loanListField;
        
        /// <remarks/>
        public MessageContext messageContext {
            get {
                return this.messageContextField;
            }
            set {
                this.messageContextField = value;
            }
        }
        
        /// <remarks/>
        public LoanFilter loanFilter {
            get {
                return this.loanFilterField;
            }
            set {
                this.loanFilterField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("loan", Namespace="http://cufxstandards.com/v3/Loan.xsd", IsNullable=false)]
        public Loan[] loanList {
            get {
                return this.loanListField;
            }
            set {
                this.loanListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Location.xsd")]
    public partial class LocationList {
        
        private Location[] locationField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("location")]
        public Location[] location {
            get {
                return this.locationField;
            }
            set {
                this.locationField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Location.xsd")]
    public partial class Location {
        
        private string locationIdField;
        
        private string nameField;
        
        private Distance distanceField;
        
        private LocationType typeField;
        
        private bool typeFieldSpecified;
        
        private Address addressField;
        
        private Phone phoneField;
        
        private string mapUrlField;
        
        private string networkField;
        
        private bool depositTakingField;
        
        private bool depositTakingFieldSpecified;
        
        private LocationHours[] locationHoursListField;
        
        private ValuePair[] servicesListField;
        
        private ValuePair[] additionalDataListField;
        
        /// <remarks/>
        public string locationId {
            get {
                return this.locationIdField;
            }
            set {
                this.locationIdField = value;
            }
        }
        
        /// <remarks/>
        public string name {
            get {
                return this.nameField;
            }
            set {
                this.nameField = value;
            }
        }
        
        /// <remarks/>
        public Distance distance {
            get {
                return this.distanceField;
            }
            set {
                this.distanceField = value;
            }
        }
        
        /// <remarks/>
        public LocationType type {
            get {
                return this.typeField;
            }
            set {
                this.typeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool typeSpecified {
            get {
                return this.typeFieldSpecified;
            }
            set {
                this.typeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public Address address {
            get {
                return this.addressField;
            }
            set {
                this.addressField = value;
            }
        }
        
        /// <remarks/>
        public Phone phone {
            get {
                return this.phoneField;
            }
            set {
                this.phoneField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="anyURI")]
        public string mapUrl {
            get {
                return this.mapUrlField;
            }
            set {
                this.mapUrlField = value;
            }
        }
        
        /// <remarks/>
        public string network {
            get {
                return this.networkField;
            }
            set {
                this.networkField = value;
            }
        }
        
        /// <remarks/>
        public bool depositTaking {
            get {
                return this.depositTakingField;
            }
            set {
                this.depositTakingField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool depositTakingSpecified {
            get {
                return this.depositTakingFieldSpecified;
            }
            set {
                this.depositTakingFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("locationHours", IsNullable=false)]
        public LocationHours[] locationHoursList {
            get {
                return this.locationHoursListField;
            }
            set {
                this.locationHoursListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("servicesList", IsNullable=false)]
        public ValuePair[] servicesList {
            get {
                return this.servicesListField;
            }
            set {
                this.servicesListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("additionalDataList", IsNullable=false)]
        public ValuePair[] additionalDataList {
            get {
                return this.additionalDataListField;
            }
            set {
                this.additionalDataListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Common.xsd")]
    public partial class Distance {
        
        private string unitField;
        
        private decimal valueField;
        
        /// <remarks/>
        public string unit {
            get {
                return this.unitField;
            }
            set {
                this.unitField = value;
            }
        }
        
        /// <remarks/>
        public decimal value {
            get {
                return this.valueField;
            }
            set {
                this.valueField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(AnonymousType=true, Namespace="http://cufxstandards.com/v3/Location.xsd")]
    public enum LocationType {
        
        /// <remarks/>
        ATM,
        
        /// <remarks/>
        SharedATM,
        
        /// <remarks/>
        Branch,
        
        /// <remarks/>
        SharedBranch,
        
        /// <remarks/>
        Kiosk,
        
        /// <remarks/>
        Other,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Location.xsd")]
    public partial class LocationHours {
        
        private LocationHoursLocationHoursType locationHoursTypeField;
        
        private bool locationHoursTypeFieldSpecified;
        
        private string descriptionField;
        
        private DayOfTheWeek dayOfTheWeekField;
        
        private bool dayOfTheWeekFieldSpecified;
        
        private System.DateTime openTimeField;
        
        private bool openTimeFieldSpecified;
        
        private System.DateTime closeTimeField;
        
        private bool closeTimeFieldSpecified;
        
        private bool closedAllDayField;
        
        private bool closedAllDayFieldSpecified;
        
        private bool openAllDayField;
        
        private bool openAllDayFieldSpecified;
        
        /// <remarks/>
        public LocationHoursLocationHoursType locationHoursType {
            get {
                return this.locationHoursTypeField;
            }
            set {
                this.locationHoursTypeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool locationHoursTypeSpecified {
            get {
                return this.locationHoursTypeFieldSpecified;
            }
            set {
                this.locationHoursTypeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string description {
            get {
                return this.descriptionField;
            }
            set {
                this.descriptionField = value;
            }
        }
        
        /// <remarks/>
        public DayOfTheWeek dayOfTheWeek {
            get {
                return this.dayOfTheWeekField;
            }
            set {
                this.dayOfTheWeekField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool dayOfTheWeekSpecified {
            get {
                return this.dayOfTheWeekFieldSpecified;
            }
            set {
                this.dayOfTheWeekFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="time")]
        public System.DateTime openTime {
            get {
                return this.openTimeField;
            }
            set {
                this.openTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool openTimeSpecified {
            get {
                return this.openTimeFieldSpecified;
            }
            set {
                this.openTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="time")]
        public System.DateTime closeTime {
            get {
                return this.closeTimeField;
            }
            set {
                this.closeTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool closeTimeSpecified {
            get {
                return this.closeTimeFieldSpecified;
            }
            set {
                this.closeTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public bool closedAllDay {
            get {
                return this.closedAllDayField;
            }
            set {
                this.closedAllDayField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool closedAllDaySpecified {
            get {
                return this.closedAllDayFieldSpecified;
            }
            set {
                this.closedAllDayFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public bool openAllDay {
            get {
                return this.openAllDayField;
            }
            set {
                this.openAllDayField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool openAllDaySpecified {
            get {
                return this.openAllDayFieldSpecified;
            }
            set {
                this.openAllDayFieldSpecified = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(AnonymousType=true, Namespace="http://cufxstandards.com/v3/Location.xsd")]
    public enum LocationHoursLocationHoursType {
        
        /// <remarks/>
        Lobby,
        
        /// <remarks/>
        DriveUp,
        
        /// <remarks/>
        ATM,
        
        /// <remarks/>
        Other,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/LocationFilter.xsd")]
    public partial class LocationFilter {
        
        private string locationIdField;
        
        private LocationFilterType typeField;
        
        private bool typeFieldSpecified;
        
        private Address searchFromAddressField;
        
        private bool depositTakingField;
        
        private bool depositTakingFieldSpecified;
        
        private string maxNumberOfResultsField;
        
        private Distance maxDistanceField;
        
        /// <remarks/>
        public string locationId {
            get {
                return this.locationIdField;
            }
            set {
                this.locationIdField = value;
            }
        }
        
        /// <remarks/>
        public LocationFilterType type {
            get {
                return this.typeField;
            }
            set {
                this.typeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool typeSpecified {
            get {
                return this.typeFieldSpecified;
            }
            set {
                this.typeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public Address searchFromAddress {
            get {
                return this.searchFromAddressField;
            }
            set {
                this.searchFromAddressField = value;
            }
        }
        
        /// <remarks/>
        public bool depositTaking {
            get {
                return this.depositTakingField;
            }
            set {
                this.depositTakingField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool depositTakingSpecified {
            get {
                return this.depositTakingFieldSpecified;
            }
            set {
                this.depositTakingFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="integer")]
        public string maxNumberOfResults {
            get {
                return this.maxNumberOfResultsField;
            }
            set {
                this.maxNumberOfResultsField = value;
            }
        }
        
        /// <remarks/>
        public Distance maxDistance {
            get {
                return this.maxDistanceField;
            }
            set {
                this.maxDistanceField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(AnonymousType=true, Namespace="http://cufxstandards.com/v3/LocationFilter.xsd")]
    public enum LocationFilterType {
        
        /// <remarks/>
        ATM,
        
        /// <remarks/>
        SharedATM,
        
        /// <remarks/>
        Branch,
        
        /// <remarks/>
        SharedBranch,
        
        /// <remarks/>
        Kiosk,
        
        /// <remarks/>
        Other,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/LocationMessage.xsd")]
    public partial class LocationMessage {
        
        private MessageContext messageContextField;
        
        private LocationFilter locationFilterField;
        
        private Location[] locationListField;
        
        /// <remarks/>
        public MessageContext messageContext {
            get {
                return this.messageContextField;
            }
            set {
                this.messageContextField = value;
            }
        }
        
        /// <remarks/>
        public LocationFilter locationFilter {
            get {
                return this.locationFilterField;
            }
            set {
                this.locationFilterField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("location", Namespace="http://cufxstandards.com/v3/Location.xsd", IsNullable=false)]
        public Location[] locationList {
            get {
                return this.locationListField;
            }
            set {
                this.locationListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/MicroDepositFunding.xsd")]
    public partial class MicroDepositFundingList {
        
        private MicroDepositFunding[] microDepositFundingField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("microDepositFunding")]
        public MicroDepositFunding[] microDepositFunding {
            get {
                return this.microDepositFundingField;
            }
            set {
                this.microDepositFundingField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/MicroDepositFunding.xsd")]
    public partial class MicroDepositFunding {
        
        private string microDepositFundingIdField;
        
        private string relationshipIdField;
        
        private string partyIdField;
        
        private string sourceAccountField;
        
        private string externalAccountIDField;
        
        private string externalRoutingNumberField;
        
        private MicroDeposit[] microDepositListField;
        
        private ValuePair[] customDataField;
        
        /// <remarks/>
        public string microDepositFundingId {
            get {
                return this.microDepositFundingIdField;
            }
            set {
                this.microDepositFundingIdField = value;
            }
        }
        
        /// <remarks/>
        public string relationshipId {
            get {
                return this.relationshipIdField;
            }
            set {
                this.relationshipIdField = value;
            }
        }
        
        /// <remarks/>
        public string partyId {
            get {
                return this.partyIdField;
            }
            set {
                this.partyIdField = value;
            }
        }
        
        /// <remarks/>
        public string sourceAccount {
            get {
                return this.sourceAccountField;
            }
            set {
                this.sourceAccountField = value;
            }
        }
        
        /// <remarks/>
        public string externalAccountID {
            get {
                return this.externalAccountIDField;
            }
            set {
                this.externalAccountIDField = value;
            }
        }
        
        /// <remarks/>
        public string externalRoutingNumber {
            get {
                return this.externalRoutingNumberField;
            }
            set {
                this.externalRoutingNumberField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("microDeposit", IsNullable=false)]
        public MicroDeposit[] microDepositList {
            get {
                return this.microDepositListField;
            }
            set {
                this.microDepositListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("valuePair", Namespace="http://cufxstandards.com/v3/Common.xsd", IsNullable=false)]
        public ValuePair[] customData {
            get {
                return this.customDataField;
            }
            set {
                this.customDataField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/MicroDepositFunding.xsd")]
    public partial class MicroDeposit {
        
        private string microDepositIdField;
        
        private Money microDepositAmountField;
        
        private string microDepositConfirmationCodeField;
        
        /// <remarks/>
        public string microDepositId {
            get {
                return this.microDepositIdField;
            }
            set {
                this.microDepositIdField = value;
            }
        }
        
        /// <remarks/>
        public Money microDepositAmount {
            get {
                return this.microDepositAmountField;
            }
            set {
                this.microDepositAmountField = value;
            }
        }
        
        /// <remarks/>
        public string microDepositConfirmationCode {
            get {
                return this.microDepositConfirmationCodeField;
            }
            set {
                this.microDepositConfirmationCodeField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/MicroDepositFundingFilter.xsd")]
    public partial class MicroDepositFundingFilter {
        
        private string[] microDepositFundingIdListField;
        
        private string[] microDepositIdListField;
        
        private string[] partyIdListField;
        
        private string[] relationshipIdListField;
        
        private string[] sourceAccountIdListField;
        
        private string[] externalAccountIdListField;
        
        private string microDepositConfirmationCodeField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("microDepositFundingId", Namespace="http://cufxstandards.com/v3/MicroDepositFunding.xsd", IsNullable=false)]
        public string[] MicroDepositFundingIdList {
            get {
                return this.microDepositFundingIdListField;
            }
            set {
                this.microDepositFundingIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("microDepositId", Namespace="http://cufxstandards.com/v3/MicroDepositFunding.xsd", IsNullable=false)]
        public string[] MicroDepositIdList {
            get {
                return this.microDepositIdListField;
            }
            set {
                this.microDepositIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("partyId", Namespace="http://cufxstandards.com/v3/Party.xsd", IsNullable=false)]
        public string[] partyIdList {
            get {
                return this.partyIdListField;
            }
            set {
                this.partyIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("relationshipId", Namespace="http://cufxstandards.com/v3/Relationship.xsd", IsNullable=false)]
        public string[] relationshipIdList {
            get {
                return this.relationshipIdListField;
            }
            set {
                this.relationshipIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("accountId", Namespace="http://cufxstandards.com/v3/Account.xsd", IsNullable=false)]
        public string[] sourceAccountIdList {
            get {
                return this.sourceAccountIdListField;
            }
            set {
                this.sourceAccountIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("accountId", Namespace="http://cufxstandards.com/v3/Account.xsd", IsNullable=false)]
        public string[] externalAccountIdList {
            get {
                return this.externalAccountIdListField;
            }
            set {
                this.externalAccountIdListField = value;
            }
        }
        
        /// <remarks/>
        public string microDepositConfirmationCode {
            get {
                return this.microDepositConfirmationCodeField;
            }
            set {
                this.microDepositConfirmationCodeField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/MicroDepositFundingMessage.xsd")]
    public partial class MicroDepositFundingMessage {
        
        private MessageContext messageContextField;
        
        private MicroDepositFundingFilter microDepositFundingFilterField;
        
        private MicroDepositFunding[] microDepositFundingListField;
        
        /// <remarks/>
        public MessageContext messageContext {
            get {
                return this.messageContextField;
            }
            set {
                this.messageContextField = value;
            }
        }
        
        /// <remarks/>
        public MicroDepositFundingFilter microDepositFundingFilter {
            get {
                return this.microDepositFundingFilterField;
            }
            set {
                this.microDepositFundingFilterField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("microDepositFunding", Namespace="http://cufxstandards.com/v3/MicroDepositFunding.xsd", IsNullable=false)]
        public MicroDepositFunding[] microDepositFundingList {
            get {
                return this.microDepositFundingListField;
            }
            set {
                this.microDepositFundingListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/NotesFilter.xsd")]
    public partial class NotesFilter {
        
        private string[] partyIdListField;
        
        private string[] accountIdListField;
        
        private string[] relationshipIdListField;
        
        private System.DateTime noteCreatedStartDateTimeField;
        
        private bool noteCreatedStartDateTimeFieldSpecified;
        
        private System.DateTime noteCreatedEndDateTimeField;
        
        private bool noteCreatedEndDateTimeFieldSpecified;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("partyId", Namespace="http://cufxstandards.com/v3/Party.xsd", IsNullable=false)]
        public string[] partyIdList {
            get {
                return this.partyIdListField;
            }
            set {
                this.partyIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("accountId", Namespace="http://cufxstandards.com/v3/Account.xsd", IsNullable=false)]
        public string[] accountIdList {
            get {
                return this.accountIdListField;
            }
            set {
                this.accountIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("relationshipId", Namespace="http://cufxstandards.com/v3/Relationship.xsd", IsNullable=false)]
        public string[] relationshipIdList {
            get {
                return this.relationshipIdListField;
            }
            set {
                this.relationshipIdListField = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime noteCreatedStartDateTime {
            get {
                return this.noteCreatedStartDateTimeField;
            }
            set {
                this.noteCreatedStartDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool noteCreatedStartDateTimeSpecified {
            get {
                return this.noteCreatedStartDateTimeFieldSpecified;
            }
            set {
                this.noteCreatedStartDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime noteCreatedEndDateTime {
            get {
                return this.noteCreatedEndDateTimeField;
            }
            set {
                this.noteCreatedEndDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool noteCreatedEndDateTimeSpecified {
            get {
                return this.noteCreatedEndDateTimeFieldSpecified;
            }
            set {
                this.noteCreatedEndDateTimeFieldSpecified = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/NoteMessage.xsd")]
    public partial class NoteMessage {
        
        private MessageContext messageContextField;
        
        private NotesFilter notesFilterField;
        
        private Note[] noteListField;
        
        /// <remarks/>
        public MessageContext messageContext {
            get {
                return this.messageContextField;
            }
            set {
                this.messageContextField = value;
            }
        }
        
        /// <remarks/>
        public NotesFilter notesFilter {
            get {
                return this.notesFilterField;
            }
            set {
                this.notesFilterField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("note", Namespace="http://cufxstandards.com/v3/Common.xsd", IsNullable=false)]
        public Note[] noteList {
            get {
                return this.noteListField;
            }
            set {
                this.noteListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/OverdraftPriorityList.xsd")]
    public partial class OverdraftPriorityList {
        
        private OverdraftPriority[] overdraftPriorityField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("overdraftPriority")]
        public OverdraftPriority[] overdraftPriority {
            get {
                return this.overdraftPriorityField;
            }
            set {
                this.overdraftPriorityField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/OverdraftPriorityList.xsd")]
    public partial class OverdraftPriority {
        
        private string accountIdField;
        
        private OverdraftPriorityAccount[] overdraftPriorityAccountField;
        
        /// <remarks/>
        public string accountId {
            get {
                return this.accountIdField;
            }
            set {
                this.accountIdField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("overdraftPriorityAccount")]
        public OverdraftPriorityAccount[] overdraftPriorityAccount {
            get {
                return this.overdraftPriorityAccountField;
            }
            set {
                this.overdraftPriorityAccountField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/OverdraftPriorityList.xsd")]
    public partial class OverdraftPriorityAccount {
        
        private string overdraftPriorityIdField;
        
        private int priorityField;
        
        private string accountIdField;
        
        /// <remarks/>
        public string overdraftPriorityId {
            get {
                return this.overdraftPriorityIdField;
            }
            set {
                this.overdraftPriorityIdField = value;
            }
        }
        
        /// <remarks/>
        public int priority {
            get {
                return this.priorityField;
            }
            set {
                this.priorityField = value;
            }
        }
        
        /// <remarks/>
        public string accountId {
            get {
                return this.accountIdField;
            }
            set {
                this.accountIdField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/OverdraftPriorityListFilter.xsd")]
    public partial class OverdraftPriorityListFilter {
        
        private string accountIdField;
        
        private string[] overdraftPriorityIdListField;
        
        private int[] priorityListField;
        
        private string[] overdraftFromAccountIdListField;
        
        /// <remarks/>
        public string accountId {
            get {
                return this.accountIdField;
            }
            set {
                this.accountIdField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("overdraftPriorityId", Namespace="http://cufxstandards.com/v3/OverdraftPriorityList.xsd", IsNullable=false)]
        public string[] overdraftPriorityIdList {
            get {
                return this.overdraftPriorityIdListField;
            }
            set {
                this.overdraftPriorityIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("priority", Namespace="http://cufxstandards.com/v3/OverdraftPriorityList.xsd", IsNullable=false)]
        public int[] priorityList {
            get {
                return this.priorityListField;
            }
            set {
                this.priorityListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("accountId", Namespace="http://cufxstandards.com/v3/Account.xsd", IsNullable=false)]
        public string[] overdraftFromAccountIdList {
            get {
                return this.overdraftFromAccountIdListField;
            }
            set {
                this.overdraftFromAccountIdListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/OverdraftPriorityMessage.xsd")]
    public partial class OverdraftPriorityMessage {
        
        private MessageContext messageContextField;
        
        private OverdraftPriorityListFilter overdraftPriorityFilterField;
        
        private OverdraftPriority[] overdraftPriorityListField;
        
        /// <remarks/>
        public MessageContext messageContext {
            get {
                return this.messageContextField;
            }
            set {
                this.messageContextField = value;
            }
        }
        
        /// <remarks/>
        public OverdraftPriorityListFilter overdraftPriorityFilter {
            get {
                return this.overdraftPriorityFilterField;
            }
            set {
                this.overdraftPriorityFilterField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("overdraftPriority", Namespace="http://cufxstandards.com/v3/OverdraftPriorityList.xsd", IsNullable=false)]
        public OverdraftPriority[] overdraftPriorityList {
            get {
                return this.overdraftPriorityListField;
            }
            set {
                this.overdraftPriorityListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Party.xsd")]
    public partial class PartyList {
        
        private Party[] partyField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("party")]
        public Party[] party {
            get {
                return this.partyField;
            }
            set {
                this.partyField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/PartyAssociation.xsd")]
    public partial class PartyAssociationList {
        
        private PartyAssociation[] partyAssociationField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("partyAssociation")]
        public PartyAssociation[] partyAssociation {
            get {
                return this.partyAssociationField;
            }
            set {
                this.partyAssociationField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/PartyAssociation.xsd")]
    public partial class PartyAssociation {
        
        private string partyAssociationIdField;
        
        private string parentPartyIdField;
        
        private string childPartyIdField;
        
        private PartyAssociationType partyAssociationTypeField;
        
        private ValuePair[] customDataField;
        
        /// <remarks/>
        public string partyAssociationId {
            get {
                return this.partyAssociationIdField;
            }
            set {
                this.partyAssociationIdField = value;
            }
        }
        
        /// <remarks/>
        public string parentPartyId {
            get {
                return this.parentPartyIdField;
            }
            set {
                this.parentPartyIdField = value;
            }
        }
        
        /// <remarks/>
        public string childPartyId {
            get {
                return this.childPartyIdField;
            }
            set {
                this.childPartyIdField = value;
            }
        }
        
        /// <remarks/>
        public PartyAssociationType partyAssociationType {
            get {
                return this.partyAssociationTypeField;
            }
            set {
                this.partyAssociationTypeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("valuePair", Namespace="http://cufxstandards.com/v3/Common.xsd", IsNullable=false)]
        public ValuePair[] customData {
            get {
                return this.customDataField;
            }
            set {
                this.customDataField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/PartyAssociation.xsd")]
    public partial class PartyAssociationType {
        
        private Partner partnerField;
        
        private OfficerManager officerManagerField;
        
        private Agent agentField;
        
        private NextOfKin nextOfKinField;
        
        private Trustor trustorField;
        
        /// <remarks/>
        public Partner partner {
            get {
                return this.partnerField;
            }
            set {
                this.partnerField = value;
            }
        }
        
        /// <remarks/>
        public OfficerManager officerManager {
            get {
                return this.officerManagerField;
            }
            set {
                this.officerManagerField = value;
            }
        }
        
        /// <remarks/>
        public Agent agent {
            get {
                return this.agentField;
            }
            set {
                this.agentField = value;
            }
        }
        
        /// <remarks/>
        public NextOfKin nextOfKin {
            get {
                return this.nextOfKinField;
            }
            set {
                this.nextOfKinField = value;
            }
        }
        
        /// <remarks/>
        public Trustor trustor {
            get {
                return this.trustorField;
            }
            set {
                this.trustorField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/PartyAssociation.xsd")]
    public partial class Partner {
        
        private PartnerQualifer qualifierField;
        
        /// <remarks/>
        public PartnerQualifer qualifier {
            get {
                return this.qualifierField;
            }
            set {
                this.qualifierField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/PartyAssociation.xsd")]
    public enum PartnerQualifer {
        
        /// <remarks/>
        GeneralPartner,
        
        /// <remarks/>
        LimitedPartner,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/PartyAssociation.xsd")]
    public partial class OfficerManager {
        
        private OfficerManagerQualifer qualifierField;
        
        /// <remarks/>
        public OfficerManagerQualifer qualifier {
            get {
                return this.qualifierField;
            }
            set {
                this.qualifierField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/PartyAssociation.xsd")]
    public enum OfficerManagerQualifer {
        
        /// <remarks/>
        BoardOfDirectors,
        
        /// <remarks/>
        CxO,
        
        /// <remarks/>
        FinancialOfficer,
        
        /// <remarks/>
        President,
        
        /// <remarks/>
        SoleProprietor,
        
        /// <remarks/>
        VicePresident,
        
        /// <remarks/>
        ExecutiveDirector,
        
        /// <remarks/>
        Director,
        
        /// <remarks/>
        Manager,
        
        /// <remarks/>
        Treasurer,
        
        /// <remarks/>
        Secretary,
        
        /// <remarks/>
        OtherManager,
        
        /// <remarks/>
        NonManager,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/PartyAssociation.xsd")]
    public partial class NextOfKin {
        
        private NextOfKinQualifer qualifierField;
        
        /// <remarks/>
        public NextOfKinQualifer qualifier {
            get {
                return this.qualifierField;
            }
            set {
                this.qualifierField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/PartyAssociation.xsd")]
    public enum NextOfKinQualifer {
        
        /// <remarks/>
        NotSpecified,
        
        /// <remarks/>
        Parent,
        
        /// <remarks/>
        Spouse,
        
        /// <remarks/>
        FirstCousin,
        
        /// <remarks/>
        Child,
        
        /// <remarks/>
        Sibling,
        
        /// <remarks/>
        GrandChild,
        
        /// <remarks/>
        GreatGrandchild,
        
        /// <remarks/>
        NieceNephew,
        
        /// <remarks/>
        AuntUncle,
        
        /// <remarks/>
        GreatGrandparent,
        
        /// <remarks/>
        GreatNieceNephew,
        
        /// <remarks/>
        GreatAuntUncle,
        
        /// <remarks/>
        GreatGreatGrandparent,
        
        /// <remarks/>
        GreatGreatGrandChild,
        
        /// <remarks/>
        FirstCousinOnceRemoved,
        
        /// <remarks/>
        GreatGrandAuntUncle,
        
        /// <remarks/>
        GreatGreatGreatGrandChild,
        
        /// <remarks/>
        GreatGreatGreatGrandParent,
        
        /// <remarks/>
        FirstCousinTwiceRemoved,
        
        /// <remarks/>
        SecondCousin,
        
        /// <remarks/>
        Other,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/PartyAssociation.xsd")]
    public partial class Trustor {
        
        private TrustorQualifer qualifierField;
        
        /// <remarks/>
        public TrustorQualifer qualifier {
            get {
                return this.qualifierField;
            }
            set {
                this.qualifierField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/PartyAssociation.xsd")]
    public enum TrustorQualifer {
        
        /// <remarks/>
        ProvidesFunds,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/PartyAssociationFilter.xsd")]
    public partial class PartyAssociationFilter {
        
        private string[] partyAssociationIdListField;
        
        private string[] parentPartyIdListField;
        
        private string[] childPartyIdListField;
        
        private PartyAssociationType[] partyAssociationTypeListField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("partyAssociationId", Namespace="http://cufxstandards.com/v3/PartyAssociation.xsd", IsNullable=false)]
        public string[] partyAssociationIdList {
            get {
                return this.partyAssociationIdListField;
            }
            set {
                this.partyAssociationIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("partyId", Namespace="http://cufxstandards.com/v3/Party.xsd", IsNullable=false)]
        public string[] parentPartyIdList {
            get {
                return this.parentPartyIdListField;
            }
            set {
                this.parentPartyIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("partyId", Namespace="http://cufxstandards.com/v3/Party.xsd", IsNullable=false)]
        public string[] childPartyIdList {
            get {
                return this.childPartyIdListField;
            }
            set {
                this.childPartyIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("partyAssociationType", Namespace="http://cufxstandards.com/v3/PartyAssociation.xsd", IsNullable=false)]
        public PartyAssociationType[] partyAssociationTypeList {
            get {
                return this.partyAssociationTypeListField;
            }
            set {
                this.partyAssociationTypeListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/PartyAssociationMessage.xsd")]
    public partial class PartyAssociationMessage {
        
        private MessageContext messageContextField;
        
        private PartyAssociationFilter partyAssociationFilterField;
        
        private PartyAssociation[] partyAssociationListField;
        
        /// <remarks/>
        public MessageContext messageContext {
            get {
                return this.messageContextField;
            }
            set {
                this.messageContextField = value;
            }
        }
        
        /// <remarks/>
        public PartyAssociationFilter partyAssociationFilter {
            get {
                return this.partyAssociationFilterField;
            }
            set {
                this.partyAssociationFilterField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("partyAssociation", Namespace="http://cufxstandards.com/v3/PartyAssociation.xsd", IsNullable=false)]
        public PartyAssociation[] partyAssociationList {
            get {
                return this.partyAssociationListField;
            }
            set {
                this.partyAssociationListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/PartyFilter.xsd")]
    public partial class PartyFilter {
        
        private string[] partyIdListField;
        
        private string[] relationshipIdListField;
        
        private string[] contactIdListField;
        
        private string[] accountIdListField;
        
        private string[] taxIdListField;
        
        private PartyType[] partyTypeListField;
        
        private bool includeNotesFlagField;
        
        private bool includeNotesFlagFieldSpecified;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("partyId", Namespace="http://cufxstandards.com/v3/Party.xsd", IsNullable=false)]
        public string[] partyIdList {
            get {
                return this.partyIdListField;
            }
            set {
                this.partyIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("relationshipId", Namespace="http://cufxstandards.com/v3/Relationship.xsd", IsNullable=false)]
        public string[] relationshipIdList {
            get {
                return this.relationshipIdListField;
            }
            set {
                this.relationshipIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("contactId", Namespace="http://cufxstandards.com/v3/Contact.xsd", IsNullable=false)]
        public string[] contactIdList {
            get {
                return this.contactIdListField;
            }
            set {
                this.contactIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("accountId", Namespace="http://cufxstandards.com/v3/Account.xsd", IsNullable=false)]
        public string[] accountIdList {
            get {
                return this.accountIdListField;
            }
            set {
                this.accountIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("taxId", Namespace="http://cufxstandards.com/v3/Party.xsd", DataType="token", IsNullable=false)]
        public string[] taxIdList {
            get {
                return this.taxIdListField;
            }
            set {
                this.taxIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("partyType", Namespace="http://cufxstandards.com/v3/Party.xsd", IsNullable=false)]
        public PartyType[] partyTypeList {
            get {
                return this.partyTypeListField;
            }
            set {
                this.partyTypeListField = value;
            }
        }
        
        /// <remarks/>
        public bool includeNotesFlag {
            get {
                return this.includeNotesFlagField;
            }
            set {
                this.includeNotesFlagField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool includeNotesFlagSpecified {
            get {
                return this.includeNotesFlagFieldSpecified;
            }
            set {
                this.includeNotesFlagFieldSpecified = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/PartyMessage.xsd")]
    public partial class PartyMessage {
        
        private MessageContext messageContextField;
        
        private PartyFilter partyFilterField;
        
        private Party[] partyListField;
        
        /// <remarks/>
        public MessageContext messageContext {
            get {
                return this.messageContextField;
            }
            set {
                this.messageContextField = value;
            }
        }
        
        /// <remarks/>
        public PartyFilter partyFilter {
            get {
                return this.partyFilterField;
            }
            set {
                this.partyFilterField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("party", Namespace="http://cufxstandards.com/v3/Party.xsd", IsNullable=false)]
        public Party[] partyList {
            get {
                return this.partyListField;
            }
            set {
                this.partyListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/PermissionList.xsd")]
    public partial class PermissionList {
        
        private Permission[] permissionField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("permission")]
        public Permission[] permission {
            get {
                return this.permissionField;
            }
            set {
                this.permissionField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/PermissionList.xsd")]
    public partial class Permission {
        
        private Actor actorField;
        
        private string actionField;
        
        private PermissionResource resourceField;
        
        private MaxLimits maxLimitsField;
        
        /// <remarks/>
        public Actor actor {
            get {
                return this.actorField;
            }
            set {
                this.actorField = value;
            }
        }
        
        /// <remarks/>
        public string action {
            get {
                return this.actionField;
            }
            set {
                this.actionField = value;
            }
        }
        
        /// <remarks/>
        public PermissionResource resource {
            get {
                return this.resourceField;
            }
            set {
                this.resourceField = value;
            }
        }
        
        /// <remarks/>
        public MaxLimits maxLimits {
            get {
                return this.maxLimitsField;
            }
            set {
                this.maxLimitsField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(AnonymousType=true, Namespace="http://cufxstandards.com/v3/PermissionList.xsd")]
    public partial class PermissionResource {
        
        private string itemField;
        
        private ItemChoiceType2 itemElementNameField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("accountId", typeof(string))]
        [System.Xml.Serialization.XmlElementAttribute("cardId", typeof(string))]
        [System.Xml.Serialization.XmlElementAttribute("fiUserId", typeof(string))]
        [System.Xml.Serialization.XmlElementAttribute("productId", typeof(string))]
        [System.Xml.Serialization.XmlElementAttribute("relationshipId", typeof(string))]
        [System.Xml.Serialization.XmlChoiceIdentifierAttribute("ItemElementName")]
        public string Item {
            get {
                return this.itemField;
            }
            set {
                this.itemField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public ItemChoiceType2 ItemElementName {
            get {
                return this.itemElementNameField;
            }
            set {
                this.itemElementNameField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/PermissionList.xsd", IncludeInSchema=false)]
    public enum ItemChoiceType2 {
        
        /// <remarks/>
        accountId,
        
        /// <remarks/>
        cardId,
        
        /// <remarks/>
        fiUserId,
        
        /// <remarks/>
        productId,
        
        /// <remarks/>
        relationshipId,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/PermissionList.xsd")]
    public partial class MaxLimits {
        
        private PermissionMaximums dailyMaxField;
        
        private PermissionMaximums weeklyMaxField;
        
        private PermissionMaximums monthlyMaxField;
        
        /// <remarks/>
        public PermissionMaximums dailyMax {
            get {
                return this.dailyMaxField;
            }
            set {
                this.dailyMaxField = value;
            }
        }
        
        /// <remarks/>
        public PermissionMaximums weeklyMax {
            get {
                return this.weeklyMaxField;
            }
            set {
                this.weeklyMaxField = value;
            }
        }
        
        /// <remarks/>
        public PermissionMaximums monthlyMax {
            get {
                return this.monthlyMaxField;
            }
            set {
                this.monthlyMaxField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/PermissionList.xsd")]
    public partial class PermissionMaximums {
        
        private Money maxAmountField;
        
        private string maxNumberField;
        
        private Money rollingAmountField;
        
        private string rollingNumberField;
        
        /// <remarks/>
        public Money maxAmount {
            get {
                return this.maxAmountField;
            }
            set {
                this.maxAmountField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="integer")]
        public string maxNumber {
            get {
                return this.maxNumberField;
            }
            set {
                this.maxNumberField = value;
            }
        }
        
        /// <remarks/>
        public Money rollingAmount {
            get {
                return this.rollingAmountField;
            }
            set {
                this.rollingAmountField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="integer")]
        public string rollingNumber {
            get {
                return this.rollingNumberField;
            }
            set {
                this.rollingNumberField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/PermissionListFilter.xsd")]
    public partial class PermissionListFilter {
        
        private Actor[] actorsField;
        
        private string actionListField;
        
        private PermissionListFilterResources resourcesField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("actors")]
        public Actor[] actors {
            get {
                return this.actorsField;
            }
            set {
                this.actorsField = value;
            }
        }
        
        /// <remarks/>
        public string actionList {
            get {
                return this.actionListField;
            }
            set {
                this.actionListField = value;
            }
        }
        
        /// <remarks/>
        public PermissionListFilterResources resources {
            get {
                return this.resourcesField;
            }
            set {
                this.resourcesField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(AnonymousType=true, Namespace="http://cufxstandards.com/v3/PermissionListFilter.xsd")]
    public partial class PermissionListFilterResources {
        
        private string[] accountIdListField;
        
        private string[] fiUserIdListField;
        
        private string[] cardIdListField;
        
        private string[] relationshipIdListField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("accountId", Namespace="http://cufxstandards.com/v3/Account.xsd", IsNullable=false)]
        public string[] accountIdList {
            get {
                return this.accountIdListField;
            }
            set {
                this.accountIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("fiUserId", Namespace="http://cufxstandards.com/v3/CredentialGroup.xsd", IsNullable=false)]
        public string[] fiUserIdList {
            get {
                return this.fiUserIdListField;
            }
            set {
                this.fiUserIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("cardId", Namespace="http://cufxstandards.com/v3/Card.xsd", IsNullable=false)]
        public string[] cardIdList {
            get {
                return this.cardIdListField;
            }
            set {
                this.cardIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("relationshipId", Namespace="http://cufxstandards.com/v3/Relationship.xsd", IsNullable=false)]
        public string[] relationshipIdList {
            get {
                return this.relationshipIdListField;
            }
            set {
                this.relationshipIdListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/PermissionListMessage.xsd")]
    public partial class PermissionListMessage {
        
        private MessageContext messageContextField;
        
        private PermissionListFilter permissionListFilterField;
        
        private Permission[] permissionListField;
        
        /// <remarks/>
        public MessageContext messageContext {
            get {
                return this.messageContextField;
            }
            set {
                this.messageContextField = value;
            }
        }
        
        /// <remarks/>
        public PermissionListFilter permissionListFilter {
            get {
                return this.permissionListFilterField;
            }
            set {
                this.permissionListFilterField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("permission", Namespace="http://cufxstandards.com/v3/PermissionList.xsd", IsNullable=false)]
        public Permission[] permissionList {
            get {
                return this.permissionListField;
            }
            set {
                this.permissionListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Preference.xsd")]
    public partial class PreferenceList {
        
        private Preference[] preferenceField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("preference", IsNullable=true)]
        public Preference[] preference {
            get {
                return this.preferenceField;
            }
            set {
                this.preferenceField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Preference.xsd")]
    public partial class Preference {
        
        private string preferenceIdField;
        
        private string[] partyIdListField;
        
        private string[] relationshipIdListField;
        
        private string[] accountIdListField;
        
        private string[] cardIdListField;
        
        private string[] contactIdListField;
        
        private DeliveryChannel deliveryChannelField;
        
        private bool deliveryChannelFieldSpecified;
        
        private PreferenceType preferenceTypeField;
        
        private bool preferenceTypeFieldSpecified;
        
        private SubType subTypeField;
        
        private System.DateTime lastChangedDateTimeField;
        
        private bool lastChangedDateTimeFieldSpecified;
        
        private PreferenceStatus preferenceStatusField;
        
        private string valueField;
        
        private string alertCustomTextField;
        
        private string[] whereToContactIdListField;
        
        private bool actionableAlertField;
        
        private bool actionableAlertFieldSpecified;
        
        private string widgetNameField;
        
        private ValuePair[] customDataField;
        
        public Preference() {
            this.preferenceStatusField = PreferenceStatus.Active;
        }
        
        /// <remarks/>
        public string preferenceId {
            get {
                return this.preferenceIdField;
            }
            set {
                this.preferenceIdField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("partyId", Namespace="http://cufxstandards.com/v3/Party.xsd", IsNullable=false)]
        public string[] partyIdList {
            get {
                return this.partyIdListField;
            }
            set {
                this.partyIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("relationshipId", Namespace="http://cufxstandards.com/v3/Relationship.xsd", IsNullable=false)]
        public string[] relationshipIdList {
            get {
                return this.relationshipIdListField;
            }
            set {
                this.relationshipIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("accountId", Namespace="http://cufxstandards.com/v3/Account.xsd", IsNullable=false)]
        public string[] accountIdList {
            get {
                return this.accountIdListField;
            }
            set {
                this.accountIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("cardId", Namespace="http://cufxstandards.com/v3/Card.xsd", IsNullable=false)]
        public string[] cardIdList {
            get {
                return this.cardIdListField;
            }
            set {
                this.cardIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("contactId", Namespace="http://cufxstandards.com/v3/Contact.xsd", IsNullable=false)]
        public string[] contactIdList {
            get {
                return this.contactIdListField;
            }
            set {
                this.contactIdListField = value;
            }
        }
        
        /// <remarks/>
        public DeliveryChannel deliveryChannel {
            get {
                return this.deliveryChannelField;
            }
            set {
                this.deliveryChannelField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool deliveryChannelSpecified {
            get {
                return this.deliveryChannelFieldSpecified;
            }
            set {
                this.deliveryChannelFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public PreferenceType preferenceType {
            get {
                return this.preferenceTypeField;
            }
            set {
                this.preferenceTypeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool preferenceTypeSpecified {
            get {
                return this.preferenceTypeFieldSpecified;
            }
            set {
                this.preferenceTypeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public SubType subType {
            get {
                return this.subTypeField;
            }
            set {
                this.subTypeField = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime lastChangedDateTime {
            get {
                return this.lastChangedDateTimeField;
            }
            set {
                this.lastChangedDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool lastChangedDateTimeSpecified {
            get {
                return this.lastChangedDateTimeFieldSpecified;
            }
            set {
                this.lastChangedDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.ComponentModel.DefaultValueAttribute(PreferenceStatus.Active)]
        public PreferenceStatus preferenceStatus {
            get {
                return this.preferenceStatusField;
            }
            set {
                this.preferenceStatusField = value;
            }
        }
        
        /// <remarks/>
        public string value {
            get {
                return this.valueField;
            }
            set {
                this.valueField = value;
            }
        }
        
        /// <remarks/>
        public string alertCustomText {
            get {
                return this.alertCustomTextField;
            }
            set {
                this.alertCustomTextField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("contactId", Namespace="http://cufxstandards.com/v3/Contact.xsd", IsNullable=false)]
        public string[] whereToContactIdList {
            get {
                return this.whereToContactIdListField;
            }
            set {
                this.whereToContactIdListField = value;
            }
        }
        
        /// <remarks/>
        public bool actionableAlert {
            get {
                return this.actionableAlertField;
            }
            set {
                this.actionableAlertField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool actionableAlertSpecified {
            get {
                return this.actionableAlertFieldSpecified;
            }
            set {
                this.actionableAlertFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string widgetName {
            get {
                return this.widgetNameField;
            }
            set {
                this.widgetNameField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("valuePair", Namespace="http://cufxstandards.com/v3/Common.xsd", IsNullable=false)]
        public ValuePair[] customData {
            get {
                return this.customDataField;
            }
            set {
                this.customDataField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Preference.xsd")]
    public enum PreferenceType {
        
        /// <remarks/>
        CourtesyPayment,
        
        /// <remarks/>
        EAlert,
        
        /// <remarks/>
        EStatement,
        
        /// <remarks/>
        ETaxForms,
        
        /// <remarks/>
        ENotice,
        
        /// <remarks/>
        EReceipt,
        
        /// <remarks/>
        EnrolledInTextBanking,
        
        /// <remarks/>
        Newsletter,
        
        /// <remarks/>
        Communication,
        
        /// <remarks/>
        Marketing,
        
        /// <remarks/>
        ContactHours,
        
        /// <remarks/>
        Language,
        
        /// <remarks/>
        WebsiteFormatStylesheet,
        
        /// <remarks/>
        MobileSiteFormatStylesheet,
        
        /// <remarks/>
        EmailFormat,
        
        /// <remarks/>
        Timeout,
        
        /// <remarks/>
        Widget,
        
        /// <remarks/>
        Other,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Preference.xsd")]
    public partial class SubType {
        
        private object itemField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("eAlertSubType", typeof(EAlertSubType))]
        [System.Xml.Serialization.XmlElementAttribute("emailFormatSubType", typeof(EmailFormatSubType))]
        [System.Xml.Serialization.XmlElementAttribute("widgetSubType", typeof(WidgetSubType))]
        public object Item {
            get {
                return this.itemField;
            }
            set {
                this.itemField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Preference.xsd")]
    public enum EAlertSubType {
        
        /// <remarks/>
        AddressChanged,
        
        /// <remarks/>
        ApproachingCreditLimit,
        
        /// <remarks/>
        AtmWithdrawalExceeds,
        
        /// <remarks/>
        AtmDepositExceeds,
        
        /// <remarks/>
        AutomatedTransferExceeds,
        
        /// <remarks/>
        BalanceFellBelow,
        
        /// <remarks/>
        BalanceExceeds,
        
        /// <remarks/>
        BillPayExceeds,
        
        /// <remarks/>
        BillPayPayeeAdded,
        
        /// <remarks/>
        BillPayPaymentRejected,
        
        /// <remarks/>
        BillPayPaymentPaid,
        
        /// <remarks/>
        CheckNumberWithinRangeHasCleared,
        
        /// <remarks/>
        CheckWithdrawalExceeds,
        
        /// <remarks/>
        CourtesyPayExceeds,
        
        /// <remarks/>
        CourtesyPayFallsBelow,
        
        /// <remarks/>
        CreditCardAuthorizationDeclined,
        
        /// <remarks/>
        CreditCardAuthorizationExceeds,
        
        /// <remarks/>
        CreditCardFuelDispenserAuthorizationExceeds,
        
        /// <remarks/>
        CreditCardInternationalAuthorizationExceeds,
        
        /// <remarks/>
        CreditCardOnlineAuthorizationExceeds,
        
        /// <remarks/>
        CreditCardOutOfStateAuthorizationExceeds,
        
        /// <remarks/>
        CreditCardRefundExceeds,
        
        /// <remarks/>
        DebitCardAuthorizationDeclined,
        
        /// <remarks/>
        DebitCardAuthorizationExceeds,
        
        /// <remarks/>
        DebitCardFuelDispenserAuthorizationExceeds,
        
        /// <remarks/>
        DebitCardInternationalAuthorizationExceeds,
        
        /// <remarks/>
        DebitCardOnlineAuthorizationExceeds,
        
        /// <remarks/>
        DebitCardOutOfStateAuthorizationExceeds,
        
        /// <remarks/>
        DebitCardRefundExceeds,
        
        /// <remarks/>
        DepositedFundsHaveBeenReturned,
        
        /// <remarks/>
        DirectDepositExceeds,
        
        /// <remarks/>
        DirectDepositFellBelow,
        
        /// <remarks/>
        EmailAddressChanged,
        
        /// <remarks/>
        ExternalTransferExceeds,
        
        /// <remarks/>
        PotentialFraud,
        
        /// <remarks/>
        IncomingWireExceeds,
        
        /// <remarks/>
        InsufficientFundsToPayCheck,
        
        /// <remarks/>
        HoldPlacedOnAccountExceeded,
        
        /// <remarks/>
        HoldThatExceededWasRemovedFromAccount,
        
        /// <remarks/>
        PhoneNumberChanged,
        
        /// <remarks/>
        LoginOccurred,
        
        /// <remarks/>
        LoginFailed,
        
        /// <remarks/>
        NameChangeAttempted,
        
        /// <remarks/>
        NameChangeOccurred,
        
        /// <remarks/>
        OutgoingWireExceeds,
        
        /// <remarks/>
        PasswordResetSuccess,
        
        /// <remarks/>
        PasswordResetFailure,
        
        /// <remarks/>
        PaymentDueInXDays,
        
        /// <remarks/>
        PinChanged,
        
        /// <remarks/>
        PreferenceChanged,
        
        /// <remarks/>
        ScheduledMessage,
        
        /// <remarks/>
        ScheduledPaymentHasFailed,
        
        /// <remarks/>
        ScheduledPaymentHasStopped,
        
        /// <remarks/>
        SpecificCheckNumberHasCleared,
        
        /// <remarks/>
        SsnChanged,
        
        /// <remarks/>
        SystemNotification,
        
        /// <remarks/>
        StatementAvailable,
        
        /// <remarks/>
        TaxFormAvailable,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Preference.xsd")]
    public enum EmailFormatSubType {
        
        /// <remarks/>
        Html,
        
        /// <remarks/>
        Text,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Preference.xsd")]
    public enum WidgetSubType {
        
        /// <remarks/>
        DisplayStatus,
        
        /// <remarks/>
        DisplayOrder,
        
        /// <remarks/>
        ShortCutKey,
        
        /// <remarks/>
        DisplayPage,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Preference.xsd")]
    public enum PreferenceStatus {
        
        /// <remarks/>
        Template,
        
        /// <remarks/>
        Active,
        
        /// <remarks/>
        Inactive,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/PreferenceFilter.xsd")]
    public partial class PreferenceFilter {
        
        private string[] preferenceIdListField;
        
        private string[] partyIdListField;
        
        private string[] relationshipIdListField;
        
        private string[] accountIdListField;
        
        private string[] cardIdListField;
        
        private string[] contactIdListField;
        
        private PreferenceType[] preferenceTypeListField;
        
        private SubType[] preferenceSubTypeListField;
        
        private PreferenceStatus[] preferenceStatusListField;
        
        private string[] widgetNameListField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("preferenceId", Namespace="http://cufxstandards.com/v3/Preference.xsd", IsNullable=false)]
        public string[] preferenceIdList {
            get {
                return this.preferenceIdListField;
            }
            set {
                this.preferenceIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("partyId", Namespace="http://cufxstandards.com/v3/Party.xsd", IsNullable=false)]
        public string[] partyIdList {
            get {
                return this.partyIdListField;
            }
            set {
                this.partyIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("relationshipId", Namespace="http://cufxstandards.com/v3/Relationship.xsd", IsNullable=false)]
        public string[] relationshipIdList {
            get {
                return this.relationshipIdListField;
            }
            set {
                this.relationshipIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("accountId", Namespace="http://cufxstandards.com/v3/Account.xsd", IsNullable=false)]
        public string[] accountIdList {
            get {
                return this.accountIdListField;
            }
            set {
                this.accountIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("cardId", Namespace="http://cufxstandards.com/v3/Card.xsd", IsNullable=false)]
        public string[] cardIdList {
            get {
                return this.cardIdListField;
            }
            set {
                this.cardIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("contactId", Namespace="http://cufxstandards.com/v3/Contact.xsd", IsNullable=false)]
        public string[] contactIdList {
            get {
                return this.contactIdListField;
            }
            set {
                this.contactIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("preferenceType", Namespace="http://cufxstandards.com/v3/Preference.xsd", IsNullable=false)]
        public PreferenceType[] preferenceTypeList {
            get {
                return this.preferenceTypeListField;
            }
            set {
                this.preferenceTypeListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("subType", Namespace="http://cufxstandards.com/v3/Preference.xsd", IsNullable=false)]
        public SubType[] preferenceSubTypeList {
            get {
                return this.preferenceSubTypeListField;
            }
            set {
                this.preferenceSubTypeListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("preferenceStatus", Namespace="http://cufxstandards.com/v3/Preference.xsd", IsNullable=false)]
        public PreferenceStatus[] preferenceStatusList {
            get {
                return this.preferenceStatusListField;
            }
            set {
                this.preferenceStatusListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("widgetName", Namespace="http://cufxstandards.com/v3/Preference.xsd", IsNullable=false)]
        public string[] widgetNameList {
            get {
                return this.widgetNameListField;
            }
            set {
                this.widgetNameListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/PreferenceMessage.xsd")]
    public partial class PreferenceMessage {
        
        private MessageContext messageContextField;
        
        private PreferenceFilter preferenceFilterField;
        
        private Preference[] preferenceListField;
        
        /// <remarks/>
        public MessageContext messageContext {
            get {
                return this.messageContextField;
            }
            set {
                this.messageContextField = value;
            }
        }
        
        /// <remarks/>
        public PreferenceFilter preferenceFilter {
            get {
                return this.preferenceFilterField;
            }
            set {
                this.preferenceFilterField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("preference", Namespace="http://cufxstandards.com/v3/Preference.xsd")]
        public Preference[] preferenceList {
            get {
                return this.preferenceListField;
            }
            set {
                this.preferenceListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/ProductOffering.xsd")]
    public partial class ProductOfferingList {
        
        private ProductOffering[] productOfferingField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("productOffering")]
        public ProductOffering[] productOffering {
            get {
                return this.productOfferingField;
            }
            set {
                this.productOfferingField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/ProductOffering.xsd")]
    public partial class ProductOffering {
        
        private string productIdField;
        
        private AccountType productTypeField;
        
        private string productSubTypeField;
        
        private ValuePair[] systemProductKeysField;
        
        private string descriptionField;
        
        private bool pointsRewardsProgramField;
        
        private bool pointsRewardsProgramFieldSpecified;
        
        private InterestRate[] interestRateListField;
        
        private string brandField;
        
        private ValuePair[] customDataField;
        
        /// <remarks/>
        public string productId {
            get {
                return this.productIdField;
            }
            set {
                this.productIdField = value;
            }
        }
        
        /// <remarks/>
        public AccountType productType {
            get {
                return this.productTypeField;
            }
            set {
                this.productTypeField = value;
            }
        }
        
        /// <remarks/>
        public string productSubType {
            get {
                return this.productSubTypeField;
            }
            set {
                this.productSubTypeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("valuePair", Namespace="http://cufxstandards.com/v3/Common.xsd", IsNullable=false)]
        public ValuePair[] systemProductKeys {
            get {
                return this.systemProductKeysField;
            }
            set {
                this.systemProductKeysField = value;
            }
        }
        
        /// <remarks/>
        public string description {
            get {
                return this.descriptionField;
            }
            set {
                this.descriptionField = value;
            }
        }
        
        /// <remarks/>
        public bool pointsRewardsProgram {
            get {
                return this.pointsRewardsProgramField;
            }
            set {
                this.pointsRewardsProgramField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool pointsRewardsProgramSpecified {
            get {
                return this.pointsRewardsProgramFieldSpecified;
            }
            set {
                this.pointsRewardsProgramFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("interestRate", IsNullable=false)]
        public InterestRate[] interestRateList {
            get {
                return this.interestRateListField;
            }
            set {
                this.interestRateListField = value;
            }
        }
        
        /// <remarks/>
        public string brand {
            get {
                return this.brandField;
            }
            set {
                this.brandField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("valuePair", Namespace="http://cufxstandards.com/v3/Common.xsd", IsNullable=false)]
        public ValuePair[] customData {
            get {
                return this.customDataField;
            }
            set {
                this.customDataField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/ProductOffering.xsd")]
    public partial class InterestRate {
        
        private string interestRateIdField;
        
        private bool interestPointsRewardsProgramField;
        
        private bool interestPointsRewardsProgramFieldSpecified;
        
        private bool introductoryRateField;
        
        private bool introductoryRateFieldSpecified;
        
        private bool balanceConsolidationRateField;
        
        private bool balanceConsolidationRateFieldSpecified;
        
        private bool riskRateField;
        
        private bool riskRateFieldSpecified;
        
        private string riskRateClassificationMinimumField;
        
        private string riskRateClassificationMaximumField;
        
        private string interestRateDescriptionField;
        
        private string interestRateCriteriaField;
        
        private Money minimumBalanceField;
        
        private Money maximumBalanceField;
        
        private string termField;
        
        private System.DateTime effectiveDateTimeField;
        
        private bool effectiveDateTimeFieldSpecified;
        
        private System.DateTime expirationDateTimeField;
        
        private bool expirationDateTimeFieldSpecified;
        
        private decimal interestRateField;
        
        private bool interestRateFieldSpecified;
        
        /// <remarks/>
        public string interestRateId {
            get {
                return this.interestRateIdField;
            }
            set {
                this.interestRateIdField = value;
            }
        }
        
        /// <remarks/>
        public bool interestPointsRewardsProgram {
            get {
                return this.interestPointsRewardsProgramField;
            }
            set {
                this.interestPointsRewardsProgramField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool interestPointsRewardsProgramSpecified {
            get {
                return this.interestPointsRewardsProgramFieldSpecified;
            }
            set {
                this.interestPointsRewardsProgramFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public bool introductoryRate {
            get {
                return this.introductoryRateField;
            }
            set {
                this.introductoryRateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool introductoryRateSpecified {
            get {
                return this.introductoryRateFieldSpecified;
            }
            set {
                this.introductoryRateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public bool balanceConsolidationRate {
            get {
                return this.balanceConsolidationRateField;
            }
            set {
                this.balanceConsolidationRateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool balanceConsolidationRateSpecified {
            get {
                return this.balanceConsolidationRateFieldSpecified;
            }
            set {
                this.balanceConsolidationRateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public bool riskRate {
            get {
                return this.riskRateField;
            }
            set {
                this.riskRateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool riskRateSpecified {
            get {
                return this.riskRateFieldSpecified;
            }
            set {
                this.riskRateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string riskRateClassificationMinimum {
            get {
                return this.riskRateClassificationMinimumField;
            }
            set {
                this.riskRateClassificationMinimumField = value;
            }
        }
        
        /// <remarks/>
        public string riskRateClassificationMaximum {
            get {
                return this.riskRateClassificationMaximumField;
            }
            set {
                this.riskRateClassificationMaximumField = value;
            }
        }
        
        /// <remarks/>
        public string interestRateDescription {
            get {
                return this.interestRateDescriptionField;
            }
            set {
                this.interestRateDescriptionField = value;
            }
        }
        
        /// <remarks/>
        public string interestRateCriteria {
            get {
                return this.interestRateCriteriaField;
            }
            set {
                this.interestRateCriteriaField = value;
            }
        }
        
        /// <remarks/>
        public Money minimumBalance {
            get {
                return this.minimumBalanceField;
            }
            set {
                this.minimumBalanceField = value;
            }
        }
        
        /// <remarks/>
        public Money maximumBalance {
            get {
                return this.maximumBalanceField;
            }
            set {
                this.maximumBalanceField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="duration")]
        public string term {
            get {
                return this.termField;
            }
            set {
                this.termField = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime effectiveDateTime {
            get {
                return this.effectiveDateTimeField;
            }
            set {
                this.effectiveDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool effectiveDateTimeSpecified {
            get {
                return this.effectiveDateTimeFieldSpecified;
            }
            set {
                this.effectiveDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime expirationDateTime {
            get {
                return this.expirationDateTimeField;
            }
            set {
                this.expirationDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool expirationDateTimeSpecified {
            get {
                return this.expirationDateTimeFieldSpecified;
            }
            set {
                this.expirationDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public decimal interestRate {
            get {
                return this.interestRateField;
            }
            set {
                this.interestRateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool interestRateSpecified {
            get {
                return this.interestRateFieldSpecified;
            }
            set {
                this.interestRateFieldSpecified = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/ProductOfferingFilter.xsd")]
    public partial class ProductOfferingFilter {
        
        private string[] productIdListField;
        
        private AccountType[] productTypeListField;
        
        private string[] productSubTypeListField;
        
        private bool introductoryRateField;
        
        private bool introductoryRateFieldSpecified;
        
        private bool balanceConsolidationRateField;
        
        private bool balanceConsolidationRateFieldSpecified;
        
        private bool riskRateField;
        
        private bool riskRateFieldSpecified;
        
        private System.DateTime interestDateTimeField;
        
        private bool interestDateTimeFieldSpecified;
        
        private string minTermField;
        
        private string maxTermField;
        
        private bool pointsRewardsProgramField;
        
        private bool pointsRewardsProgramFieldSpecified;
        
        private string brandField;
        
        private string riskRateClassificationField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("productId", Namespace="http://cufxstandards.com/v3/ProductOffering.xsd")]
        public string[] productIdList {
            get {
                return this.productIdListField;
            }
            set {
                this.productIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("productType", Namespace="http://cufxstandards.com/v3/ProductOffering.xsd", IsNullable=false)]
        public AccountType[] productTypeList {
            get {
                return this.productTypeListField;
            }
            set {
                this.productTypeListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("productSubType", Namespace="http://cufxstandards.com/v3/ProductOffering.xsd")]
        public string[] productSubTypeList {
            get {
                return this.productSubTypeListField;
            }
            set {
                this.productSubTypeListField = value;
            }
        }
        
        /// <remarks/>
        public bool introductoryRate {
            get {
                return this.introductoryRateField;
            }
            set {
                this.introductoryRateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool introductoryRateSpecified {
            get {
                return this.introductoryRateFieldSpecified;
            }
            set {
                this.introductoryRateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public bool balanceConsolidationRate {
            get {
                return this.balanceConsolidationRateField;
            }
            set {
                this.balanceConsolidationRateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool balanceConsolidationRateSpecified {
            get {
                return this.balanceConsolidationRateFieldSpecified;
            }
            set {
                this.balanceConsolidationRateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public bool riskRate {
            get {
                return this.riskRateField;
            }
            set {
                this.riskRateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool riskRateSpecified {
            get {
                return this.riskRateFieldSpecified;
            }
            set {
                this.riskRateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime interestDateTime {
            get {
                return this.interestDateTimeField;
            }
            set {
                this.interestDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool interestDateTimeSpecified {
            get {
                return this.interestDateTimeFieldSpecified;
            }
            set {
                this.interestDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="duration")]
        public string minTerm {
            get {
                return this.minTermField;
            }
            set {
                this.minTermField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="duration")]
        public string maxTerm {
            get {
                return this.maxTermField;
            }
            set {
                this.maxTermField = value;
            }
        }
        
        /// <remarks/>
        public bool pointsRewardsProgram {
            get {
                return this.pointsRewardsProgramField;
            }
            set {
                this.pointsRewardsProgramField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool pointsRewardsProgramSpecified {
            get {
                return this.pointsRewardsProgramFieldSpecified;
            }
            set {
                this.pointsRewardsProgramFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string brand {
            get {
                return this.brandField;
            }
            set {
                this.brandField = value;
            }
        }
        
        /// <remarks/>
        public string riskRateClassification {
            get {
                return this.riskRateClassificationField;
            }
            set {
                this.riskRateClassificationField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/ProductOfferingMessage.xsd")]
    public partial class ProductOfferingMessage {
        
        private MessageContext messageContextField;
        
        private ProductOfferingFilter productOfferingFilterField;
        
        private ProductOffering[] productOfferingListField;
        
        /// <remarks/>
        public MessageContext messageContext {
            get {
                return this.messageContextField;
            }
            set {
                this.messageContextField = value;
            }
        }
        
        /// <remarks/>
        public ProductOfferingFilter productOfferingFilter {
            get {
                return this.productOfferingFilterField;
            }
            set {
                this.productOfferingFilterField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("productOffering", Namespace="http://cufxstandards.com/v3/ProductOffering.xsd", IsNullable=false)]
        public ProductOffering[] productOfferingList {
            get {
                return this.productOfferingListField;
            }
            set {
                this.productOfferingListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/ProductServiceRequest.xsd")]
    public partial class ProductServiceRequestList {
        
        private ProductServiceRequest[] productServiceRequestField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("productServiceRequest", IsNullable=true)]
        public ProductServiceRequest[] productServiceRequest {
            get {
                return this.productServiceRequestField;
            }
            set {
                this.productServiceRequestField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/ProductServiceRequest.xsd")]
    public partial class ProductServiceRequest {
        
        private string productServiceRequestIdField;
        
        private Request[] requestListField;
        
        private string sourceField;
        
        private ProductServiceStatus statusField;
        
        private bool statusFieldSpecified;
        
        private System.DateTime createDateTimeField;
        
        private bool createDateTimeFieldSpecified;
        
        private RelatedToGroups productServiceRequestRelatedToField;
        
        private Note[] productServiceRequestNoteListField;
        
        private ValuePair[] productServiceRequestCustomDataField;
        
        /// <remarks/>
        public string productServiceRequestId {
            get {
                return this.productServiceRequestIdField;
            }
            set {
                this.productServiceRequestIdField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("request", IsNullable=false)]
        public Request[] requestList {
            get {
                return this.requestListField;
            }
            set {
                this.requestListField = value;
            }
        }
        
        /// <remarks/>
        public string source {
            get {
                return this.sourceField;
            }
            set {
                this.sourceField = value;
            }
        }
        
        /// <remarks/>
        public ProductServiceStatus status {
            get {
                return this.statusField;
            }
            set {
                this.statusField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool statusSpecified {
            get {
                return this.statusFieldSpecified;
            }
            set {
                this.statusFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime createDateTime {
            get {
                return this.createDateTimeField;
            }
            set {
                this.createDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool createDateTimeSpecified {
            get {
                return this.createDateTimeFieldSpecified;
            }
            set {
                this.createDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public RelatedToGroups productServiceRequestRelatedTo {
            get {
                return this.productServiceRequestRelatedToField;
            }
            set {
                this.productServiceRequestRelatedToField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("note", Namespace="http://cufxstandards.com/v3/Common.xsd", IsNullable=false)]
        public Note[] productServiceRequestNoteList {
            get {
                return this.productServiceRequestNoteListField;
            }
            set {
                this.productServiceRequestNoteListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("valuePair", Namespace="http://cufxstandards.com/v3/Common.xsd", IsNullable=false)]
        public ValuePair[] productServiceRequestCustomData {
            get {
                return this.productServiceRequestCustomDataField;
            }
            set {
                this.productServiceRequestCustomDataField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/ProductServiceRequest.xsd")]
    public partial class Request {
        
        private string requestIdField;
        
        private RequestItem requestItemField;
        
        private ActivityStatus requestStatusField;
        
        private bool requestStatusFieldSpecified;
        
        private System.DateTime requestedDateTimeField;
        
        private bool requestedDateTimeFieldSpecified;
        
        private System.DateTime dueDateTimeField;
        
        private bool dueDateTimeFieldSpecified;
        
        private System.DateTime startDateTimeField;
        
        private bool startDateTimeFieldSpecified;
        
        private System.DateTime completedDateTimeField;
        
        private bool completedDateTimeFieldSpecified;
        
        private Note[] requestNoteListField;
        
        private RelatedToGroups requestRelatedToField;
        
        private Activity[] activityListField;
        
        private SecureMessage[] secureMessageListField;
        
        private string[] documentListField;
        
        private ValuePair[] requestCustomDataField;
        
        /// <remarks/>
        public string requestId {
            get {
                return this.requestIdField;
            }
            set {
                this.requestIdField = value;
            }
        }
        
        /// <remarks/>
        public RequestItem requestItem {
            get {
                return this.requestItemField;
            }
            set {
                this.requestItemField = value;
            }
        }
        
        /// <remarks/>
        public ActivityStatus requestStatus {
            get {
                return this.requestStatusField;
            }
            set {
                this.requestStatusField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool requestStatusSpecified {
            get {
                return this.requestStatusFieldSpecified;
            }
            set {
                this.requestStatusFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime requestedDateTime {
            get {
                return this.requestedDateTimeField;
            }
            set {
                this.requestedDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool requestedDateTimeSpecified {
            get {
                return this.requestedDateTimeFieldSpecified;
            }
            set {
                this.requestedDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime dueDateTime {
            get {
                return this.dueDateTimeField;
            }
            set {
                this.dueDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool dueDateTimeSpecified {
            get {
                return this.dueDateTimeFieldSpecified;
            }
            set {
                this.dueDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime startDateTime {
            get {
                return this.startDateTimeField;
            }
            set {
                this.startDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool startDateTimeSpecified {
            get {
                return this.startDateTimeFieldSpecified;
            }
            set {
                this.startDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime completedDateTime {
            get {
                return this.completedDateTimeField;
            }
            set {
                this.completedDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool completedDateTimeSpecified {
            get {
                return this.completedDateTimeFieldSpecified;
            }
            set {
                this.completedDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("note", Namespace="http://cufxstandards.com/v3/Common.xsd", IsNullable=false)]
        public Note[] requestNoteList {
            get {
                return this.requestNoteListField;
            }
            set {
                this.requestNoteListField = value;
            }
        }
        
        /// <remarks/>
        public RelatedToGroups requestRelatedTo {
            get {
                return this.requestRelatedToField;
            }
            set {
                this.requestRelatedToField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("activity", IsNullable=false)]
        public Activity[] activityList {
            get {
                return this.activityListField;
            }
            set {
                this.activityListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("secureMessage", Namespace="http://cufxstandards.com/v3/SecureMessage.xsd", IsNullable=false)]
        public SecureMessage[] secureMessageList {
            get {
                return this.secureMessageListField;
            }
            set {
                this.secureMessageListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("documentId", Namespace="http://cufxstandards.com/v3/Document.xsd", IsNullable=false)]
        public string[] documentList {
            get {
                return this.documentListField;
            }
            set {
                this.documentListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("valuePair", Namespace="http://cufxstandards.com/v3/Common.xsd", IsNullable=false)]
        public ValuePair[] requestCustomData {
            get {
                return this.requestCustomDataField;
            }
            set {
                this.requestCustomDataField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/ProductServiceRequest.xsd")]
    public partial class RequestItem {
        
        private string itemField;
        
        private ItemChoiceType3 itemElementNameField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("productOfInterest", typeof(string))]
        [System.Xml.Serialization.XmlElementAttribute("question", typeof(string))]
        [System.Xml.Serialization.XmlElementAttribute("serviceOfInterest", typeof(string))]
        [System.Xml.Serialization.XmlChoiceIdentifierAttribute("ItemElementName")]
        public string Item {
            get {
                return this.itemField;
            }
            set {
                this.itemField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public ItemChoiceType3 ItemElementName {
            get {
                return this.itemElementNameField;
            }
            set {
                this.itemElementNameField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/ProductServiceRequest.xsd", IncludeInSchema=false)]
    public enum ItemChoiceType3 {
        
        /// <remarks/>
        productOfInterest,
        
        /// <remarks/>
        question,
        
        /// <remarks/>
        serviceOfInterest,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/ProductServiceRequest.xsd")]
    public enum ActivityStatus {
        
        /// <remarks/>
        New,
        
        /// <remarks/>
        Assigned,
        
        /// <remarks/>
        InProgressActive,
        
        /// <remarks/>
        InProgressInactive,
        
        /// <remarks/>
        Completed,
        
        /// <remarks/>
        Cancelled,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/ProductServiceRequest.xsd")]
    public partial class RelatedToGroups {
        
        private string[] contactIdListField;
        
        private string[] partyIdListField;
        
        private Party[] unknownPartyListField;
        
        private string[] relationshipIdListField;
        
        private string[] accountIdListField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("contactId", Namespace="http://cufxstandards.com/v3/Contact.xsd", IsNullable=false)]
        public string[] contactIdList {
            get {
                return this.contactIdListField;
            }
            set {
                this.contactIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("partyId", Namespace="http://cufxstandards.com/v3/Party.xsd", IsNullable=false)]
        public string[] partyIdList {
            get {
                return this.partyIdListField;
            }
            set {
                this.partyIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("party", Namespace="http://cufxstandards.com/v3/Party.xsd", IsNullable=false)]
        public Party[] unknownPartyList {
            get {
                return this.unknownPartyListField;
            }
            set {
                this.unknownPartyListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("relationshipId", Namespace="http://cufxstandards.com/v3/Relationship.xsd", IsNullable=false)]
        public string[] relationshipIdList {
            get {
                return this.relationshipIdListField;
            }
            set {
                this.relationshipIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("accountId", Namespace="http://cufxstandards.com/v3/Account.xsd", IsNullable=false)]
        public string[] accountIdList {
            get {
                return this.accountIdListField;
            }
            set {
                this.accountIdListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/ProductServiceRequest.xsd")]
    public partial class Activity {
        
        private string activityIdField;
        
        private string[] previousActivityIdField;
        
        private string activityNameField;
        
        private ActivityStatus activityStatusField;
        
        private bool activityStatusFieldSpecified;
        
        private string creatorField;
        
        private System.DateTime requestedDateTimeField;
        
        private bool requestedDateTimeFieldSpecified;
        
        private System.DateTime dueDateTimeField;
        
        private bool dueDateTimeFieldSpecified;
        
        private System.DateTime startDateTimeField;
        
        private bool startDateTimeFieldSpecified;
        
        private System.DateTime completedDateTimeField;
        
        private bool completedDateTimeFieldSpecified;
        
        private Note[] activityNoteListField;
        
        private CredentialType[] credentialTypesRequiredListField;
        
        private CredentialGroup[] credentialsProvidedListField;
        
        private RelatedToGroups activityRelatedToField;
        
        private ValuePair[] activityCustomDataField;
        
        /// <remarks/>
        public string activityId {
            get {
                return this.activityIdField;
            }
            set {
                this.activityIdField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("previousActivityId")]
        public string[] previousActivityId {
            get {
                return this.previousActivityIdField;
            }
            set {
                this.previousActivityIdField = value;
            }
        }
        
        /// <remarks/>
        public string activityName {
            get {
                return this.activityNameField;
            }
            set {
                this.activityNameField = value;
            }
        }
        
        /// <remarks/>
        public ActivityStatus activityStatus {
            get {
                return this.activityStatusField;
            }
            set {
                this.activityStatusField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool activityStatusSpecified {
            get {
                return this.activityStatusFieldSpecified;
            }
            set {
                this.activityStatusFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string creator {
            get {
                return this.creatorField;
            }
            set {
                this.creatorField = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime requestedDateTime {
            get {
                return this.requestedDateTimeField;
            }
            set {
                this.requestedDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool requestedDateTimeSpecified {
            get {
                return this.requestedDateTimeFieldSpecified;
            }
            set {
                this.requestedDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime dueDateTime {
            get {
                return this.dueDateTimeField;
            }
            set {
                this.dueDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool dueDateTimeSpecified {
            get {
                return this.dueDateTimeFieldSpecified;
            }
            set {
                this.dueDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime startDateTime {
            get {
                return this.startDateTimeField;
            }
            set {
                this.startDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool startDateTimeSpecified {
            get {
                return this.startDateTimeFieldSpecified;
            }
            set {
                this.startDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime completedDateTime {
            get {
                return this.completedDateTimeField;
            }
            set {
                this.completedDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool completedDateTimeSpecified {
            get {
                return this.completedDateTimeFieldSpecified;
            }
            set {
                this.completedDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("note", Namespace="http://cufxstandards.com/v3/Common.xsd", IsNullable=false)]
        public Note[] activityNoteList {
            get {
                return this.activityNoteListField;
            }
            set {
                this.activityNoteListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("credentialType", Namespace="http://cufxstandards.com/v3/CredentialGroup.xsd", IsNullable=false)]
        public CredentialType[] credentialTypesRequiredList {
            get {
                return this.credentialTypesRequiredListField;
            }
            set {
                this.credentialTypesRequiredListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("credentialGroup", Namespace="http://cufxstandards.com/v3/CredentialGroup.xsd")]
        public CredentialGroup[] credentialsProvidedList {
            get {
                return this.credentialsProvidedListField;
            }
            set {
                this.credentialsProvidedListField = value;
            }
        }
        
        /// <remarks/>
        public RelatedToGroups activityRelatedTo {
            get {
                return this.activityRelatedToField;
            }
            set {
                this.activityRelatedToField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("valuePair", Namespace="http://cufxstandards.com/v3/Common.xsd", IsNullable=false)]
        public ValuePair[] activityCustomData {
            get {
                return this.activityCustomDataField;
            }
            set {
                this.activityCustomDataField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/SecureMessage.xsd")]
    public partial class SecureMessage {
        
        private string secureMessageIdField;
        
        private string previousSecureMessageIdField;
        
        private string threadIdField;
        
        private string subjectLineField;
        
        private SecureMessageUser[] messageFromListField;
        
        private SecureMessageUser[] messageToListField;
        
        private SecureMessageUser[] carbonCopyListField;
        
        private SecureMessageUser[] blindCarbonCopyListField;
        
        private SecureMessageUser[] replyToListField;
        
        private SecureMessageType typeField;
        
        private bool typeFieldSpecified;
        
        private SecureMessageStatus currentStatusField;
        
        private bool currentStatusFieldSpecified;
        
        private StatusLogEntry[] statusLogField;
        
        private string bodyField;
        
        private BodyFormat bodyFormatField;
        
        private bool bodyFormatFieldSpecified;
        
        private string[] documentIdListField;
        
        private System.DateTime doNotDeliverBeforeDateTimeField;
        
        private bool doNotDeliverBeforeDateTimeFieldSpecified;
        
        private System.DateTime expirationDateTimeField;
        
        private bool expirationDateTimeFieldSpecified;
        
        private string requestIdField;
        
        private ValuePair[] customDataField;
        
        /// <remarks/>
        public string secureMessageId {
            get {
                return this.secureMessageIdField;
            }
            set {
                this.secureMessageIdField = value;
            }
        }
        
        /// <remarks/>
        public string previousSecureMessageId {
            get {
                return this.previousSecureMessageIdField;
            }
            set {
                this.previousSecureMessageIdField = value;
            }
        }
        
        /// <remarks/>
        public string threadId {
            get {
                return this.threadIdField;
            }
            set {
                this.threadIdField = value;
            }
        }
        
        /// <remarks/>
        public string subjectLine {
            get {
                return this.subjectLineField;
            }
            set {
                this.subjectLineField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("secureMessageUser", IsNullable=false)]
        public SecureMessageUser[] messageFromList {
            get {
                return this.messageFromListField;
            }
            set {
                this.messageFromListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("secureMessageUser", IsNullable=false)]
        public SecureMessageUser[] messageToList {
            get {
                return this.messageToListField;
            }
            set {
                this.messageToListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("secureMessageUser", IsNullable=false)]
        public SecureMessageUser[] carbonCopyList {
            get {
                return this.carbonCopyListField;
            }
            set {
                this.carbonCopyListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("secureMessageUser", IsNullable=false)]
        public SecureMessageUser[] blindCarbonCopyList {
            get {
                return this.blindCarbonCopyListField;
            }
            set {
                this.blindCarbonCopyListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("secureMessageUser", IsNullable=false)]
        public SecureMessageUser[] replyToList {
            get {
                return this.replyToListField;
            }
            set {
                this.replyToListField = value;
            }
        }
        
        /// <remarks/>
        public SecureMessageType type {
            get {
                return this.typeField;
            }
            set {
                this.typeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool typeSpecified {
            get {
                return this.typeFieldSpecified;
            }
            set {
                this.typeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public SecureMessageStatus currentStatus {
            get {
                return this.currentStatusField;
            }
            set {
                this.currentStatusField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool currentStatusSpecified {
            get {
                return this.currentStatusFieldSpecified;
            }
            set {
                this.currentStatusFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("statusLogEntry", IsNullable=false)]
        public StatusLogEntry[] statusLog {
            get {
                return this.statusLogField;
            }
            set {
                this.statusLogField = value;
            }
        }
        
        /// <remarks/>
        public string body {
            get {
                return this.bodyField;
            }
            set {
                this.bodyField = value;
            }
        }
        
        /// <remarks/>
        public BodyFormat bodyFormat {
            get {
                return this.bodyFormatField;
            }
            set {
                this.bodyFormatField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool bodyFormatSpecified {
            get {
                return this.bodyFormatFieldSpecified;
            }
            set {
                this.bodyFormatFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("documentId", Namespace="http://cufxstandards.com/v3/Document.xsd", IsNullable=false)]
        public string[] documentIdList {
            get {
                return this.documentIdListField;
            }
            set {
                this.documentIdListField = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime doNotDeliverBeforeDateTime {
            get {
                return this.doNotDeliverBeforeDateTimeField;
            }
            set {
                this.doNotDeliverBeforeDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool doNotDeliverBeforeDateTimeSpecified {
            get {
                return this.doNotDeliverBeforeDateTimeFieldSpecified;
            }
            set {
                this.doNotDeliverBeforeDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime expirationDateTime {
            get {
                return this.expirationDateTimeField;
            }
            set {
                this.expirationDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool expirationDateTimeSpecified {
            get {
                return this.expirationDateTimeFieldSpecified;
            }
            set {
                this.expirationDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string requestId {
            get {
                return this.requestIdField;
            }
            set {
                this.requestIdField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("valuePair", Namespace="http://cufxstandards.com/v3/Common.xsd", IsNullable=false)]
        public ValuePair[] customData {
            get {
                return this.customDataField;
            }
            set {
                this.customDataField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/SecureMessage.xsd")]
    public enum SecureMessageType {
        
        /// <remarks/>
        Alert,
        
        /// <remarks/>
        GeneralMessage,
        
        /// <remarks/>
        Notice,
        
        /// <remarks/>
        ProductOffering,
        
        /// <remarks/>
        ScheduleMeeting,
        
        /// <remarks/>
        ServiceUpdate,
        
        /// <remarks/>
        SignatureRequired,
        
        /// <remarks/>
        Other,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/SecureMessage.xsd")]
    public enum SecureMessageStatus {
        
        /// <remarks/>
        Template,
        
        /// <remarks/>
        Draft,
        
        /// <remarks/>
        Outbox,
        
        /// <remarks/>
        Sent,
        
        /// <remarks/>
        Read,
        
        /// <remarks/>
        Replied,
        
        /// <remarks/>
        Forwarded,
        
        /// <remarks/>
        Archived,
        
        /// <remarks/>
        Unread,
        
        /// <remarks/>
        Failed,
        
        /// <remarks/>
        Expired,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/SecureMessage.xsd")]
    public partial class StatusLogEntry {
        
        private SecureMessageStatus statusField;
        
        private bool statusFieldSpecified;
        
        private string messageAccessProfileIdField;
        
        private System.DateTime statusDateTimeField;
        
        private bool statusDateTimeFieldSpecified;
        
        /// <remarks/>
        public SecureMessageStatus status {
            get {
                return this.statusField;
            }
            set {
                this.statusField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool statusSpecified {
            get {
                return this.statusFieldSpecified;
            }
            set {
                this.statusFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string messageAccessProfileId {
            get {
                return this.messageAccessProfileIdField;
            }
            set {
                this.messageAccessProfileIdField = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime statusDateTime {
            get {
                return this.statusDateTimeField;
            }
            set {
                this.statusDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool statusDateTimeSpecified {
            get {
                return this.statusDateTimeFieldSpecified;
            }
            set {
                this.statusDateTimeFieldSpecified = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/SecureMessage.xsd")]
    public enum BodyFormat {
        
        /// <remarks/>
        Html,
        
        /// <remarks/>
        Text,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/ProductServiceRequest.xsd")]
    public enum ProductServiceStatus {
        
        /// <remarks/>
        New,
        
        /// <remarks/>
        Assigned,
        
        /// <remarks/>
        InDiscussion,
        
        /// <remarks/>
        InProposal,
        
        /// <remarks/>
        VerbalCommitment,
        
        /// <remarks/>
        Converted,
        
        /// <remarks/>
        Lost,
        
        /// <remarks/>
        Other,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/ProductServiceRequestFilter.xsd")]
    public partial class ProductServiceRequestFilter {
        
        private string[] productServiceRequestIdListField;
        
        private string[] relationshipIdListField;
        
        private string[] accountIdListField;
        
        private string[] partyIdListField;
        
        private string[] contactIdListField;
        
        private string[] secureMessageIdListField;
        
        private ProductServiceStatus[] productServiceRequestStatusListField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("productServiceRequestId", Namespace="http://cufxstandards.com/v3/ProductServiceRequest.xsd")]
        public string[] productServiceRequestIdList {
            get {
                return this.productServiceRequestIdListField;
            }
            set {
                this.productServiceRequestIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("relationshipId", Namespace="http://cufxstandards.com/v3/Relationship.xsd", IsNullable=false)]
        public string[] relationshipIdList {
            get {
                return this.relationshipIdListField;
            }
            set {
                this.relationshipIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("accountId", Namespace="http://cufxstandards.com/v3/Account.xsd", IsNullable=false)]
        public string[] accountIdList {
            get {
                return this.accountIdListField;
            }
            set {
                this.accountIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("partyId", Namespace="http://cufxstandards.com/v3/Party.xsd", IsNullable=false)]
        public string[] partyIdList {
            get {
                return this.partyIdListField;
            }
            set {
                this.partyIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("contactId", Namespace="http://cufxstandards.com/v3/Contact.xsd", IsNullable=false)]
        public string[] contactIdList {
            get {
                return this.contactIdListField;
            }
            set {
                this.contactIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("secureMessageId", Namespace="http://cufxstandards.com/v3/SecureMessage.xsd", IsNullable=false)]
        public string[] secureMessageIdList {
            get {
                return this.secureMessageIdListField;
            }
            set {
                this.secureMessageIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("status", Namespace="http://cufxstandards.com/v3/ProductServiceRequest.xsd", IsNullable=false)]
        public ProductServiceStatus[] productServiceRequestStatusList {
            get {
                return this.productServiceRequestStatusListField;
            }
            set {
                this.productServiceRequestStatusListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/ProductServiceRequestMessage.xsd")]
    public partial class ProductServiceRequestMessage {
        
        private MessageContext messageContextField;
        
        private ProductServiceRequestFilter productServiceRequestFilterField;
        
        private ProductServiceRequest[] productServiceRequestListField;
        
        /// <remarks/>
        public MessageContext messageContext {
            get {
                return this.messageContextField;
            }
            set {
                this.messageContextField = value;
            }
        }
        
        /// <remarks/>
        public ProductServiceRequestFilter productServiceRequestFilter {
            get {
                return this.productServiceRequestFilterField;
            }
            set {
                this.productServiceRequestFilterField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("productServiceRequest", Namespace="http://cufxstandards.com/v3/ProductServiceRequest.xsd")]
        public ProductServiceRequest[] productServiceRequestList {
            get {
                return this.productServiceRequestListField;
            }
            set {
                this.productServiceRequestListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/QuestionList.xsd")]
    public partial class QuestionList {
        
        private Question[] questionField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("question")]
        public Question[] question {
            get {
                return this.questionField;
            }
            set {
                this.questionField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/QuestionList.xsd")]
    public partial class Question {
        
        private Choice[] choiceListField;
        
        private string questionIdField;
        
        private string questionTextField;
        
        private ValuePair[] customDataField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("choice", IsNullable=false)]
        public Choice[] choiceList {
            get {
                return this.choiceListField;
            }
            set {
                this.choiceListField = value;
            }
        }
        
        /// <remarks/>
        public string questionId {
            get {
                return this.questionIdField;
            }
            set {
                this.questionIdField = value;
            }
        }
        
        /// <remarks/>
        public string questionText {
            get {
                return this.questionTextField;
            }
            set {
                this.questionTextField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("valuePair", Namespace="http://cufxstandards.com/v3/Common.xsd", IsNullable=false)]
        public ValuePair[] customData {
            get {
                return this.customDataField;
            }
            set {
                this.customDataField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Ratings.xsd")]
    public partial class RatingsList {
        
        private Rating[] ratingField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("rating")]
        public Rating[] rating {
            get {
                return this.ratingField;
            }
            set {
                this.ratingField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Ratings.xsd")]
    public partial class Rating {
        
        private string reviewIdField;
        
        private System.DateTime reviewDateTimeField;
        
        private bool reviewDateTimeFieldSpecified;
        
        private string reviewerNameField;
        
        private string productCodeField;
        
        private string productNameField;
        
        private decimal overallRatingField;
        
        private bool overallRatingFieldSpecified;
        
        private bool featureReviewField;
        
        private bool featureReviewFieldSpecified;
        
        private bool recommendedField;
        
        private bool recommendedFieldSpecified;
        
        private string reviewSubjectField;
        
        private string reviewDescriptionField;
        
        private string reviewerEmailField;
        
        private string reviewerUserIdField;
        
        private string reviewerLocationField;
        
        private string reviewerAgeField;
        
        private string reviewerGenderField;
        
        private string productDescriptionField;
        
        private string productTypeField;
        
        private string productURLField;
        
        /// <remarks/>
        public string reviewId {
            get {
                return this.reviewIdField;
            }
            set {
                this.reviewIdField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime reviewDateTime {
            get {
                return this.reviewDateTimeField;
            }
            set {
                this.reviewDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool reviewDateTimeSpecified {
            get {
                return this.reviewDateTimeFieldSpecified;
            }
            set {
                this.reviewDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string reviewerName {
            get {
                return this.reviewerNameField;
            }
            set {
                this.reviewerNameField = value;
            }
        }
        
        /// <remarks/>
        public string productCode {
            get {
                return this.productCodeField;
            }
            set {
                this.productCodeField = value;
            }
        }
        
        /// <remarks/>
        public string productName {
            get {
                return this.productNameField;
            }
            set {
                this.productNameField = value;
            }
        }
        
        /// <remarks/>
        public decimal overallRating {
            get {
                return this.overallRatingField;
            }
            set {
                this.overallRatingField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool overallRatingSpecified {
            get {
                return this.overallRatingFieldSpecified;
            }
            set {
                this.overallRatingFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public bool featureReview {
            get {
                return this.featureReviewField;
            }
            set {
                this.featureReviewField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool featureReviewSpecified {
            get {
                return this.featureReviewFieldSpecified;
            }
            set {
                this.featureReviewFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public bool recommended {
            get {
                return this.recommendedField;
            }
            set {
                this.recommendedField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool recommendedSpecified {
            get {
                return this.recommendedFieldSpecified;
            }
            set {
                this.recommendedFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string reviewSubject {
            get {
                return this.reviewSubjectField;
            }
            set {
                this.reviewSubjectField = value;
            }
        }
        
        /// <remarks/>
        public string reviewDescription {
            get {
                return this.reviewDescriptionField;
            }
            set {
                this.reviewDescriptionField = value;
            }
        }
        
        /// <remarks/>
        public string reviewerEmail {
            get {
                return this.reviewerEmailField;
            }
            set {
                this.reviewerEmailField = value;
            }
        }
        
        /// <remarks/>
        public string reviewerUserId {
            get {
                return this.reviewerUserIdField;
            }
            set {
                this.reviewerUserIdField = value;
            }
        }
        
        /// <remarks/>
        public string reviewerLocation {
            get {
                return this.reviewerLocationField;
            }
            set {
                this.reviewerLocationField = value;
            }
        }
        
        /// <remarks/>
        public string reviewerAge {
            get {
                return this.reviewerAgeField;
            }
            set {
                this.reviewerAgeField = value;
            }
        }
        
        /// <remarks/>
        public string reviewerGender {
            get {
                return this.reviewerGenderField;
            }
            set {
                this.reviewerGenderField = value;
            }
        }
        
        /// <remarks/>
        public string productDescription {
            get {
                return this.productDescriptionField;
            }
            set {
                this.productDescriptionField = value;
            }
        }
        
        /// <remarks/>
        public string productType {
            get {
                return this.productTypeField;
            }
            set {
                this.productTypeField = value;
            }
        }
        
        /// <remarks/>
        public string productURL {
            get {
                return this.productURLField;
            }
            set {
                this.productURLField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/RegisteredDevice.xsd")]
    public partial class RegisteredDeviceList {
        
        private RegisteredDevice[] registeredDeviceField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("registeredDevice")]
        public RegisteredDevice[] registeredDevice {
            get {
                return this.registeredDeviceField;
            }
            set {
                this.registeredDeviceField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/RegisteredDevice.xsd")]
    public partial class RegisteredDevice {
        
        private string deviceIdField;
        
        private string[] partyIdListField;
        
        private string[] relationshipIdListField;
        
        private string[] accountIdListField;
        
        private bool encryptedFlagField;
        
        private bool encryptedFlagFieldSpecified;
        
        private bool isRegisteredFlagField;
        
        private bool isRegisteredFlagFieldSpecified;
        
        /// <remarks/>
        public string deviceId {
            get {
                return this.deviceIdField;
            }
            set {
                this.deviceIdField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("partyId", Namespace="http://cufxstandards.com/v3/Party.xsd", IsNullable=false)]
        public string[] partyIdList {
            get {
                return this.partyIdListField;
            }
            set {
                this.partyIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("relationshipId", Namespace="http://cufxstandards.com/v3/Relationship.xsd", IsNullable=false)]
        public string[] relationshipIdList {
            get {
                return this.relationshipIdListField;
            }
            set {
                this.relationshipIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("accountId", Namespace="http://cufxstandards.com/v3/Account.xsd", IsNullable=false)]
        public string[] accountIdList {
            get {
                return this.accountIdListField;
            }
            set {
                this.accountIdListField = value;
            }
        }
        
        /// <remarks/>
        public bool encryptedFlag {
            get {
                return this.encryptedFlagField;
            }
            set {
                this.encryptedFlagField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool encryptedFlagSpecified {
            get {
                return this.encryptedFlagFieldSpecified;
            }
            set {
                this.encryptedFlagFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public bool isRegisteredFlag {
            get {
                return this.isRegisteredFlagField;
            }
            set {
                this.isRegisteredFlagField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool isRegisteredFlagSpecified {
            get {
                return this.isRegisteredFlagFieldSpecified;
            }
            set {
                this.isRegisteredFlagFieldSpecified = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/RegisteredDeviceFilter.xsd")]
    public partial class RegisteredDeviceFilter {
        
        private RegisteredDeviceIdList deviceIDListField;
        
        private string[] partyIdListField;
        
        private string[] relationshipIdListField;
        
        private string[] accountIdListField;
        
        private bool encryptedFlagField;
        
        private bool encryptedFlagFieldSpecified;
        
        /// <remarks/>
        public RegisteredDeviceIdList deviceIDList {
            get {
                return this.deviceIDListField;
            }
            set {
                this.deviceIDListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("partyId", Namespace="http://cufxstandards.com/v3/Party.xsd", IsNullable=false)]
        public string[] partyIdList {
            get {
                return this.partyIdListField;
            }
            set {
                this.partyIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("relationshipId", Namespace="http://cufxstandards.com/v3/Relationship.xsd", IsNullable=false)]
        public string[] relationshipIdList {
            get {
                return this.relationshipIdListField;
            }
            set {
                this.relationshipIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("accountId", Namespace="http://cufxstandards.com/v3/Account.xsd", IsNullable=false)]
        public string[] accountIdList {
            get {
                return this.accountIdListField;
            }
            set {
                this.accountIdListField = value;
            }
        }
        
        /// <remarks/>
        public bool encryptedFlag {
            get {
                return this.encryptedFlagField;
            }
            set {
                this.encryptedFlagField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool encryptedFlagSpecified {
            get {
                return this.encryptedFlagFieldSpecified;
            }
            set {
                this.encryptedFlagFieldSpecified = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/RegisteredDevice.xsd")]
    public partial class RegisteredDeviceIdList {
        
        private string deviceIdField;
        
        /// <remarks/>
        public string deviceId {
            get {
                return this.deviceIdField;
            }
            set {
                this.deviceIdField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/RegisteredDeviceMessage.xsd")]
    public partial class RegisteredDeviceMessage {
        
        private MessageContext messageContextField;
        
        private RegisteredDeviceFilter registeredDeviceFilterField;
        
        private RegisteredDevice[] registeredDeviceListField;
        
        /// <remarks/>
        public MessageContext messageContext {
            get {
                return this.messageContextField;
            }
            set {
                this.messageContextField = value;
            }
        }
        
        /// <remarks/>
        public RegisteredDeviceFilter registeredDeviceFilter {
            get {
                return this.registeredDeviceFilterField;
            }
            set {
                this.registeredDeviceFilterField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("registeredDevice", Namespace="http://cufxstandards.com/v3/RegisteredDevice.xsd", IsNullable=false)]
        public RegisteredDevice[] registeredDeviceList {
            get {
                return this.registeredDeviceListField;
            }
            set {
                this.registeredDeviceListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Relationship.xsd")]
    public partial class RelationshipList {
        
        private Relationship[] relationshipField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("relationship")]
        public Relationship[] relationship {
            get {
                return this.relationshipField;
            }
            set {
                this.relationshipField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Relationship.xsd")]
    public partial class Relationship {
        
        private string relationshipIdField;
        
        private System.DateTime dateCreatedField;
        
        private bool dateCreatedFieldSpecified;
        
        private System.DateTime dateRelationshipEndedField;
        
        private bool dateRelationshipEndedFieldSpecified;
        
        private string rewardsCodeField;
        
        private RelationshipParty[] relationshipPartyListField;
        
        private string[] accountIdListField;
        
        private RelationshipStatus relationshipStatusField;
        
        private bool relationshipStatusFieldSpecified;
        
        /// <remarks/>
        public string relationshipId {
            get {
                return this.relationshipIdField;
            }
            set {
                this.relationshipIdField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime dateCreated {
            get {
                return this.dateCreatedField;
            }
            set {
                this.dateCreatedField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool dateCreatedSpecified {
            get {
                return this.dateCreatedFieldSpecified;
            }
            set {
                this.dateCreatedFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute(DataType="date")]
        public System.DateTime dateRelationshipEnded {
            get {
                return this.dateRelationshipEndedField;
            }
            set {
                this.dateRelationshipEndedField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool dateRelationshipEndedSpecified {
            get {
                return this.dateRelationshipEndedFieldSpecified;
            }
            set {
                this.dateRelationshipEndedFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string rewardsCode {
            get {
                return this.rewardsCodeField;
            }
            set {
                this.rewardsCodeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("relationshipParty")]
        public RelationshipParty[] relationshipPartyList {
            get {
                return this.relationshipPartyListField;
            }
            set {
                this.relationshipPartyListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("accountId", Namespace="http://cufxstandards.com/v3/Account.xsd", IsNullable=false)]
        public string[] accountIdList {
            get {
                return this.accountIdListField;
            }
            set {
                this.accountIdListField = value;
            }
        }
        
        /// <remarks/>
        public RelationshipStatus relationshipStatus {
            get {
                return this.relationshipStatusField;
            }
            set {
                this.relationshipStatusField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool relationshipStatusSpecified {
            get {
                return this.relationshipStatusFieldSpecified;
            }
            set {
                this.relationshipStatusFieldSpecified = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Relationship.xsd")]
    public partial class RelationshipParty {
        
        private string idField;
        
        private Party partyField;
        
        private PartyRelationshipType partyRelationshipTypeField;
        
        private bool ssnOverrideField;
        
        private string[] contactIdListField;
        
        private ValuePair[] customDataField;
        
        public RelationshipParty() {
            this.ssnOverrideField = false;
        }
        
        /// <remarks/>
        public string id {
            get {
                return this.idField;
            }
            set {
                this.idField = value;
            }
        }
        
        /// <remarks/>
        public Party party {
            get {
                return this.partyField;
            }
            set {
                this.partyField = value;
            }
        }
        
        /// <remarks/>
        public PartyRelationshipType partyRelationshipType {
            get {
                return this.partyRelationshipTypeField;
            }
            set {
                this.partyRelationshipTypeField = value;
            }
        }
        
        /// <remarks/>
        [System.ComponentModel.DefaultValueAttribute(false)]
        public bool ssnOverride {
            get {
                return this.ssnOverrideField;
            }
            set {
                this.ssnOverrideField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("contactId", Namespace="http://cufxstandards.com/v3/Contact.xsd", IsNullable=false)]
        public string[] contactIdList {
            get {
                return this.contactIdListField;
            }
            set {
                this.contactIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("valuePair", Namespace="http://cufxstandards.com/v3/Common.xsd", IsNullable=false)]
        public ValuePair[] customData {
            get {
                return this.customDataField;
            }
            set {
                this.customDataField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Relationship.xsd")]
    public partial class PartyRelationshipType {
        
        private Holder holderField;
        
        private Beneficiary beneficiaryField;
        
        private Agent agentField;
        
        private Payee payeeField;
        
        private Guarantor guarantorField;
        
        private CollateralGrantor collateralGrantorField;
        
        private SafeDepositBoxUser safeDepositBoxUserField;
        
        /// <remarks/>
        public Holder holder {
            get {
                return this.holderField;
            }
            set {
                this.holderField = value;
            }
        }
        
        /// <remarks/>
        public Beneficiary beneficiary {
            get {
                return this.beneficiaryField;
            }
            set {
                this.beneficiaryField = value;
            }
        }
        
        /// <remarks/>
        public Agent agent {
            get {
                return this.agentField;
            }
            set {
                this.agentField = value;
            }
        }
        
        /// <remarks/>
        public Payee payee {
            get {
                return this.payeeField;
            }
            set {
                this.payeeField = value;
            }
        }
        
        /// <remarks/>
        public Guarantor guarantor {
            get {
                return this.guarantorField;
            }
            set {
                this.guarantorField = value;
            }
        }
        
        /// <remarks/>
        public CollateralGrantor collateralGrantor {
            get {
                return this.collateralGrantorField;
            }
            set {
                this.collateralGrantorField = value;
            }
        }
        
        /// <remarks/>
        public SafeDepositBoxUser safeDepositBoxUser {
            get {
                return this.safeDepositBoxUserField;
            }
            set {
                this.safeDepositBoxUserField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Relationship.xsd")]
    public enum RelationshipStatus {
        
        /// <remarks/>
        Active,
        
        /// <remarks/>
        Closed,
        
        /// <remarks/>
        Deceased,
        
        /// <remarks/>
        Inactive,
        
        /// <remarks/>
        Prospect,
        
        /// <remarks/>
        Restricted,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/RelationshipFilter.xsd")]
    public partial class RelationshipFilter {
        
        private string[] partyIdListField;
        
        private string[] relationshipIdListField;
        
        private string[] accountIdListField;
        
        private string[] contactIdListField;
        
        private RelationshipStatus[] relationshipStatusListField;
        
        private bool includeNotesFlagField;
        
        private bool includeNotesFlagFieldSpecified;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("partyId", Namespace="http://cufxstandards.com/v3/Party.xsd", IsNullable=false)]
        public string[] partyIdList {
            get {
                return this.partyIdListField;
            }
            set {
                this.partyIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("relationshipId", Namespace="http://cufxstandards.com/v3/Relationship.xsd", IsNullable=false)]
        public string[] relationshipIdList {
            get {
                return this.relationshipIdListField;
            }
            set {
                this.relationshipIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("accountId", Namespace="http://cufxstandards.com/v3/Account.xsd", IsNullable=false)]
        public string[] accountIdList {
            get {
                return this.accountIdListField;
            }
            set {
                this.accountIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("contactId", Namespace="http://cufxstandards.com/v3/Contact.xsd", IsNullable=false)]
        public string[] contactIdList {
            get {
                return this.contactIdListField;
            }
            set {
                this.contactIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("relationshipStatus", Namespace="http://cufxstandards.com/v3/Relationship.xsd", IsNullable=false)]
        public RelationshipStatus[] relationshipStatusList {
            get {
                return this.relationshipStatusListField;
            }
            set {
                this.relationshipStatusListField = value;
            }
        }
        
        /// <remarks/>
        public bool includeNotesFlag {
            get {
                return this.includeNotesFlagField;
            }
            set {
                this.includeNotesFlagField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool includeNotesFlagSpecified {
            get {
                return this.includeNotesFlagFieldSpecified;
            }
            set {
                this.includeNotesFlagFieldSpecified = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/RelationshipMessage.xsd")]
    public partial class RelationshipMessage {
        
        private MessageContext messageContextField;
        
        private RelationshipFilter relationshipFilterField;
        
        private Relationship[] relationshipListField;
        
        /// <remarks/>
        public MessageContext messageContext {
            get {
                return this.messageContextField;
            }
            set {
                this.messageContextField = value;
            }
        }
        
        /// <remarks/>
        public RelationshipFilter relationshipFilter {
            get {
                return this.relationshipFilterField;
            }
            set {
                this.relationshipFilterField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("relationship", Namespace="http://cufxstandards.com/v3/Relationship.xsd", IsNullable=false)]
        public Relationship[] relationshipList {
            get {
                return this.relationshipListField;
            }
            set {
                this.relationshipListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/RemoteDeposit.xsd")]
    public partial class RemoteDepositList {
        
        private RemoteDeposit[] remoteDepositField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("remoteDeposit")]
        public RemoteDeposit[] remoteDeposit {
            get {
                return this.remoteDepositField;
            }
            set {
                this.remoteDepositField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/RemoteDeposit.xsd")]
    public partial class RemoteDeposit {
        
        private string remoteDepositIdField;
        
        private string clientReferenceIdField;
        
        private string vendorReferenceIdField;
        
        private string checkNumberField;
        
        private string routingTransitNumberField;
        
        private string micrCheckAccountNumberField;
        
        private Money amountField;
        
        private bool croppedField;
        
        private bool croppedFieldSpecified;
        
        private Artifact frontImageField;
        
        private Artifact backImageField;
        
        private string accountIdField;
        
        private RemoteDepositStatus statusField;
        
        private bool statusFieldSpecified;
        
        private ImageValidationIssue[] imageValidationIssueListField;
        
        private ValuePair[] customDataField;
        
        /// <remarks/>
        public string remoteDepositId {
            get {
                return this.remoteDepositIdField;
            }
            set {
                this.remoteDepositIdField = value;
            }
        }
        
        /// <remarks/>
        public string clientReferenceId {
            get {
                return this.clientReferenceIdField;
            }
            set {
                this.clientReferenceIdField = value;
            }
        }
        
        /// <remarks/>
        public string vendorReferenceId {
            get {
                return this.vendorReferenceIdField;
            }
            set {
                this.vendorReferenceIdField = value;
            }
        }
        
        /// <remarks/>
        public string checkNumber {
            get {
                return this.checkNumberField;
            }
            set {
                this.checkNumberField = value;
            }
        }
        
        /// <remarks/>
        public string routingTransitNumber {
            get {
                return this.routingTransitNumberField;
            }
            set {
                this.routingTransitNumberField = value;
            }
        }
        
        /// <remarks/>
        public string micrCheckAccountNumber {
            get {
                return this.micrCheckAccountNumberField;
            }
            set {
                this.micrCheckAccountNumberField = value;
            }
        }
        
        /// <remarks/>
        public Money amount {
            get {
                return this.amountField;
            }
            set {
                this.amountField = value;
            }
        }
        
        /// <remarks/>
        public bool cropped {
            get {
                return this.croppedField;
            }
            set {
                this.croppedField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool croppedSpecified {
            get {
                return this.croppedFieldSpecified;
            }
            set {
                this.croppedFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public Artifact frontImage {
            get {
                return this.frontImageField;
            }
            set {
                this.frontImageField = value;
            }
        }
        
        /// <remarks/>
        public Artifact backImage {
            get {
                return this.backImageField;
            }
            set {
                this.backImageField = value;
            }
        }
        
        /// <remarks/>
        public string accountId {
            get {
                return this.accountIdField;
            }
            set {
                this.accountIdField = value;
            }
        }
        
        /// <remarks/>
        public RemoteDepositStatus status {
            get {
                return this.statusField;
            }
            set {
                this.statusField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool statusSpecified {
            get {
                return this.statusFieldSpecified;
            }
            set {
                this.statusFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("imageValidationIssue", IsNullable=false)]
        public ImageValidationIssue[] imageValidationIssueList {
            get {
                return this.imageValidationIssueListField;
            }
            set {
                this.imageValidationIssueListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("valuePair", Namespace="http://cufxstandards.com/v3/Common.xsd", IsNullable=false)]
        public ValuePair[] customData {
            get {
                return this.customDataField;
            }
            set {
                this.customDataField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/RemoteDeposit.xsd")]
    public enum RemoteDepositStatus {
        
        /// <remarks/>
        ExceedsDepositLimit,
        
        /// <remarks/>
        Approved,
        
        /// <remarks/>
        UnderReview,
        
        /// <remarks/>
        DeclinedEligibility,
        
        /// <remarks/>
        DeclinedImageValidationIssue,
        
        /// <remarks/>
        DuplicateItem,
        
        /// <remarks/>
        InvalidReferenceId,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/RemoteDeposit.xsd")]
    public enum ImageValidationIssue {
        
        /// <remarks/>
        CARMismatchFailed,
        
        /// <remarks/>
        FoldedCorners,
        
        /// <remarks/>
        ExcessSkew,
        
        /// <remarks/>
        TooDark,
        
        /// <remarks/>
        TooLight,
        
        /// <remarks/>
        BelowMinSize,
        
        /// <remarks/>
        AboveMaxSize,
        
        /// <remarks/>
        UndersizedImage,
        
        /// <remarks/>
        OversizedImage,
        
        /// <remarks/>
        SpotNoise,
        
        /// <remarks/>
        DateUsability,
        
        /// <remarks/>
        PayeeUsability,
        
        /// <remarks/>
        SignatureUsability,
        
        /// <remarks/>
        PayorUsability,
        
        /// <remarks/>
        MICRUsability,
        
        /// <remarks/>
        ImageFormat,
        
        /// <remarks/>
        EndorsementUsability,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/RemoteDepositFilter.xsd")]
    public partial class RemoteDepositFilter {
        
        private string[] remoteDepositIdListField;
        
        private string[] accountIdListField;
        
        private RemoteDepositStatus[] statusListField;
        
        private RemoteDepositAction[] actionListField;
        
        private ImageValidationIssue[] imageValidationIssueListField;
        
        private System.DateTime remoteDepositStartDateTimeField;
        
        private bool remoteDepositStartDateTimeFieldSpecified;
        
        private System.DateTime remoteDepositEndDateTimeField;
        
        private bool remoteDepositEndDateTimeFieldSpecified;
        
        private IncludeImageOnRead includeImageOnReadField;
        
        private bool includeImageOnReadFieldSpecified;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("remoteDepositId", Namespace="http://cufxstandards.com/v3/RemoteDeposit.xsd", IsNullable=false)]
        public string[] remoteDepositIdList {
            get {
                return this.remoteDepositIdListField;
            }
            set {
                this.remoteDepositIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("accountId", Namespace="http://cufxstandards.com/v3/Account.xsd", IsNullable=false)]
        public string[] accountIdList {
            get {
                return this.accountIdListField;
            }
            set {
                this.accountIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("status", Namespace="http://cufxstandards.com/v3/RemoteDeposit.xsd", IsNullable=false)]
        public RemoteDepositStatus[] statusList {
            get {
                return this.statusListField;
            }
            set {
                this.statusListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("action", Namespace="http://cufxstandards.com/v3/RemoteDeposit.xsd", IsNullable=false)]
        public RemoteDepositAction[] actionList {
            get {
                return this.actionListField;
            }
            set {
                this.actionListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("imageValidationIssue", Namespace="http://cufxstandards.com/v3/RemoteDeposit.xsd", IsNullable=false)]
        public ImageValidationIssue[] imageValidationIssueList {
            get {
                return this.imageValidationIssueListField;
            }
            set {
                this.imageValidationIssueListField = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime remoteDepositStartDateTime {
            get {
                return this.remoteDepositStartDateTimeField;
            }
            set {
                this.remoteDepositStartDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool remoteDepositStartDateTimeSpecified {
            get {
                return this.remoteDepositStartDateTimeFieldSpecified;
            }
            set {
                this.remoteDepositStartDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime remoteDepositEndDateTime {
            get {
                return this.remoteDepositEndDateTimeField;
            }
            set {
                this.remoteDepositEndDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool remoteDepositEndDateTimeSpecified {
            get {
                return this.remoteDepositEndDateTimeFieldSpecified;
            }
            set {
                this.remoteDepositEndDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public IncludeImageOnRead includeImageOnRead {
            get {
                return this.includeImageOnReadField;
            }
            set {
                this.includeImageOnReadField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool includeImageOnReadSpecified {
            get {
                return this.includeImageOnReadFieldSpecified;
            }
            set {
                this.includeImageOnReadFieldSpecified = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/RemoteDeposit.xsd")]
    public enum RemoteDepositAction {
        
        /// <remarks/>
        CreateSession,
        
        /// <remarks/>
        SubmitImage,
        
        /// <remarks/>
        Commit,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/RemoteDepositFilter.xsd")]
    public enum IncludeImageOnRead {
        
        /// <remarks/>
        ArtifactIdOnly,
        
        /// <remarks/>
        FrontImage,
        
        /// <remarks/>
        BackImage,
        
        /// <remarks/>
        BothFrontBackImage,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/RemoteDepositMessage.xsd")]
    public partial class RemoteDepositMessage {
        
        private MessageContext messageContextField;
        
        private RemoteDeposit[] remoteDepositListField;
        
        /// <remarks/>
        public MessageContext messageContext {
            get {
                return this.messageContextField;
            }
            set {
                this.messageContextField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("remoteDeposit", Namespace="http://cufxstandards.com/v3/RemoteDeposit.xsd", IsNullable=false)]
        public RemoteDeposit[] remoteDepositList {
            get {
                return this.remoteDepositListField;
            }
            set {
                this.remoteDepositListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/SecureMessage.xsd")]
    public partial class SecureMessageList {
        
        private SecureMessage[] secureMessageField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("secureMessage")]
        public SecureMessage[] secureMessage {
            get {
                return this.secureMessageField;
            }
            set {
                this.secureMessageField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/SecureMessageFilter.xsd")]
    public partial class SecureMessageFilter {
        
        private string[] secureMessageIdListField;
        
        private string[] threadIdListField;
        
        private string[] subjectLineContainsListField;
        
        private SecureMessageUser[] messageFromListField;
        
        private SecureMessageUser[] messageToListField;
        
        private SecureMessageUser[] carbonCopyListField;
        
        private SecureMessageUser[] blindCarbonCopyListField;
        
        private SecureMessageUser[] replyToListField;
        
        private SecureMessageType[] secureMessageTypeListField;
        
        private SecureMessageStatus[] currentStatusListField;
        
        private SecureMessageStatus[] statusLogListField;
        
        private System.DateTime statusLogStartDateField;
        
        private bool statusLogStartDateFieldSpecified;
        
        private System.DateTime statusLogEndDateField;
        
        private bool statusLogEndDateFieldSpecified;
        
        private string bodyContainsListField;
        
        private bool returnConversationField;
        
        private bool returnConversationFieldSpecified;
        
        private string[] documentIDListField;
        
        private string[] productServiceRequestIDListField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("secureMessageId", Namespace="http://cufxstandards.com/v3/SecureMessage.xsd", IsNullable=false)]
        public string[] secureMessageIdList {
            get {
                return this.secureMessageIdListField;
            }
            set {
                this.secureMessageIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("threadId", Namespace="http://cufxstandards.com/v3/SecureMessage.xsd", IsNullable=false)]
        public string[] threadIdList {
            get {
                return this.threadIdListField;
            }
            set {
                this.threadIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("subjectLine", Namespace="http://cufxstandards.com/v3/SecureMessage.xsd", IsNullable=false)]
        public string[] subjectLineContainsList {
            get {
                return this.subjectLineContainsListField;
            }
            set {
                this.subjectLineContainsListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("secureMessageUser", Namespace="http://cufxstandards.com/v3/SecureMessage.xsd", IsNullable=false)]
        public SecureMessageUser[] messageFromList {
            get {
                return this.messageFromListField;
            }
            set {
                this.messageFromListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("secureMessageUser", Namespace="http://cufxstandards.com/v3/SecureMessage.xsd", IsNullable=false)]
        public SecureMessageUser[] messageToList {
            get {
                return this.messageToListField;
            }
            set {
                this.messageToListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("secureMessageUser", Namespace="http://cufxstandards.com/v3/SecureMessage.xsd", IsNullable=false)]
        public SecureMessageUser[] carbonCopyList {
            get {
                return this.carbonCopyListField;
            }
            set {
                this.carbonCopyListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("secureMessageUser", Namespace="http://cufxstandards.com/v3/SecureMessage.xsd", IsNullable=false)]
        public SecureMessageUser[] blindCarbonCopyList {
            get {
                return this.blindCarbonCopyListField;
            }
            set {
                this.blindCarbonCopyListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("secureMessageUser", Namespace="http://cufxstandards.com/v3/SecureMessage.xsd", IsNullable=false)]
        public SecureMessageUser[] replyToList {
            get {
                return this.replyToListField;
            }
            set {
                this.replyToListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("secureMessageType", Namespace="http://cufxstandards.com/v3/SecureMessage.xsd", IsNullable=false)]
        public SecureMessageType[] secureMessageTypeList {
            get {
                return this.secureMessageTypeListField;
            }
            set {
                this.secureMessageTypeListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("secureMessageStatus", Namespace="http://cufxstandards.com/v3/SecureMessage.xsd", IsNullable=false)]
        public SecureMessageStatus[] currentStatusList {
            get {
                return this.currentStatusListField;
            }
            set {
                this.currentStatusListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("secureMessageStatus", Namespace="http://cufxstandards.com/v3/SecureMessage.xsd", IsNullable=false)]
        public SecureMessageStatus[] statusLogList {
            get {
                return this.statusLogListField;
            }
            set {
                this.statusLogListField = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime statusLogStartDate {
            get {
                return this.statusLogStartDateField;
            }
            set {
                this.statusLogStartDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool statusLogStartDateSpecified {
            get {
                return this.statusLogStartDateFieldSpecified;
            }
            set {
                this.statusLogStartDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime statusLogEndDate {
            get {
                return this.statusLogEndDateField;
            }
            set {
                this.statusLogEndDateField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool statusLogEndDateSpecified {
            get {
                return this.statusLogEndDateFieldSpecified;
            }
            set {
                this.statusLogEndDateFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public string bodyContainsList {
            get {
                return this.bodyContainsListField;
            }
            set {
                this.bodyContainsListField = value;
            }
        }
        
        /// <remarks/>
        public bool returnConversation {
            get {
                return this.returnConversationField;
            }
            set {
                this.returnConversationField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool returnConversationSpecified {
            get {
                return this.returnConversationFieldSpecified;
            }
            set {
                this.returnConversationFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("documentId", Namespace="http://cufxstandards.com/v3/Document.xsd", IsNullable=false)]
        public string[] documentIDList {
            get {
                return this.documentIDListField;
            }
            set {
                this.documentIDListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("productServiceRequestId", Namespace="http://cufxstandards.com/v3/ProductServiceRequest.xsd")]
        public string[] productServiceRequestIDList {
            get {
                return this.productServiceRequestIDListField;
            }
            set {
                this.productServiceRequestIDListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/SecureMessageMessage.xsd")]
    public partial class SecureMessageMessage {
        
        private MessageContext messageContextField;
        
        private SecureMessageFilter secureMessageFilterField;
        
        private SecureMessage[] secureMessageListField;
        
        /// <remarks/>
        public MessageContext messageContext {
            get {
                return this.messageContextField;
            }
            set {
                this.messageContextField = value;
            }
        }
        
        /// <remarks/>
        public SecureMessageFilter secureMessageFilter {
            get {
                return this.secureMessageFilterField;
            }
            set {
                this.secureMessageFilterField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("secureMessage", Namespace="http://cufxstandards.com/v3/SecureMessage.xsd", IsNullable=false)]
        public SecureMessage[] secureMessageList {
            get {
                return this.secureMessageListField;
            }
            set {
                this.secureMessageListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/SimpleValidationRequest.xsd")]
    public partial class SimpleValidationRequestList {
        
        private SimpleValidationRequest[] simpleValidationRequestField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("simpleValidationRequest")]
        public SimpleValidationRequest[] simpleValidationRequest {
            get {
                return this.simpleValidationRequestField;
            }
            set {
                this.simpleValidationRequestField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/SimpleValidationRequest.xsd")]
    public partial class SimpleValidationRequest {
        
        private string validationTypeField;
        
        private Applicant applicantField;
        
        private ValuePair[] customDataField;
        
        /// <remarks/>
        public string validationType {
            get {
                return this.validationTypeField;
            }
            set {
                this.validationTypeField = value;
            }
        }
        
        /// <remarks/>
        public Applicant applicant {
            get {
                return this.applicantField;
            }
            set {
                this.applicantField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("valuePair", Namespace="http://cufxstandards.com/v3/Common.xsd", IsNullable=false)]
        public ValuePair[] customData {
            get {
                return this.customDataField;
            }
            set {
                this.customDataField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/SimpleValidationRequestMessage.xsd")]
    public partial class SimpleValidationRequestMessage {
        
        private MessageContext messageContextField;
        
        private SimpleValidationRequest[] simpleValidationRequestListField;
        
        /// <remarks/>
        public MessageContext messageContext {
            get {
                return this.messageContextField;
            }
            set {
                this.messageContextField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("simpleValidationRequest", Namespace="http://cufxstandards.com/v3/SimpleValidationRequest.xsd", IsNullable=false)]
        public SimpleValidationRequest[] simpleValidationRequestList {
            get {
                return this.simpleValidationRequestListField;
            }
            set {
                this.simpleValidationRequestListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/SystemStatus.xsd")]
    public partial class SystemStatusList {
        
        private SystemState[] systemStateField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("systemState")]
        public SystemState[] systemState {
            get {
                return this.systemStateField;
            }
            set {
                this.systemStateField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/SystemStatus.xsd")]
    public partial class SystemState {
        
        private string systemNameField;
        
        private string systemIdField;
        
        private SystemStatus1 systemStatusField;
        
        private bool systemStatusFieldSpecified;
        
        /// <remarks/>
        public string systemName {
            get {
                return this.systemNameField;
            }
            set {
                this.systemNameField = value;
            }
        }
        
        /// <remarks/>
        public string systemId {
            get {
                return this.systemIdField;
            }
            set {
                this.systemIdField = value;
            }
        }
        
        /// <remarks/>
        public SystemStatus1 systemStatus {
            get {
                return this.systemStatusField;
            }
            set {
                this.systemStatusField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool systemStatusSpecified {
            get {
                return this.systemStatusFieldSpecified;
            }
            set {
                this.systemStatusFieldSpecified = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(TypeName="SystemStatus", Namespace="http://cufxstandards.com/v3/SystemStatus.xsd")]
    public enum SystemStatus1 {
        
        /// <remarks/>
        Online,
        
        /// <remarks/>
        Offline,
        
        /// <remarks/>
        MemoPost,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/SystemStatusFilter.xsd")]
    public partial class SystemStatusFilter {
        
        private SystemState[] systemStatusListField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("systemState", Namespace="http://cufxstandards.com/v3/SystemStatus.xsd", IsNullable=false)]
        public SystemState[] systemStatusList {
            get {
                return this.systemStatusListField;
            }
            set {
                this.systemStatusListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/SystemStatusMessage.xsd")]
    public partial class SystemStatusMessage {
        
        private MessageContext messageContextField;
        
        private SystemStatusFilter systemStatusFilterField;
        
        private SystemState[] systemStatusListField;
        
        /// <remarks/>
        public MessageContext messageContext {
            get {
                return this.messageContextField;
            }
            set {
                this.messageContextField = value;
            }
        }
        
        /// <remarks/>
        public SystemStatusFilter systemStatusFilter {
            get {
                return this.systemStatusFilterField;
            }
            set {
                this.systemStatusFilterField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("systemState", Namespace="http://cufxstandards.com/v3/SystemStatus.xsd", IsNullable=false)]
        public SystemState[] systemStatusList {
            get {
                return this.systemStatusListField;
            }
            set {
                this.systemStatusListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Transaction.xsd")]
    public partial class TransactionList {
        
        private TransactionListTransaction[] transactionField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("transaction", IsNullable=true)]
        public TransactionListTransaction[] transaction {
            get {
                return this.transactionField;
            }
            set {
                this.transactionField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/TransactionFilter.xsd")]
    public partial class TransactionFilter {
        
        private string[] transactionIdListField;
        
        private string[] partyIdListField;
        
        private string[] accountIdListField;
        
        private TransactionType transactionTypeField;
        
        private bool transactionTypeFieldSpecified;
        
        private TransactionStatus[] transactionStatusListField;
        
        private Money transactionMinAmountField;
        
        private Money transactionMaxAmountField;
        
        private string descriptionContainsField;
        
        private string[] checkNumberListField;
        
        private System.DateTime transactionEffectiveStartDateTimeField;
        
        private bool transactionEffectiveStartDateTimeFieldSpecified;
        
        private System.DateTime transactionEffectiveEndDateTimeField;
        
        private bool transactionEffectiveEndDateTimeFieldSpecified;
        
        private System.DateTime transactionPostedStartDateTimeField;
        
        private bool transactionPostedStartDateTimeFieldSpecified;
        
        private System.DateTime transactionPostedEndDateTimeField;
        
        private bool transactionPostedEndDateTimeFieldSpecified;
        
        private string[] categoryListField;
        
        private TransactionSource[] sourceListField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("transactionId", Namespace="http://cufxstandards.com/v3/Transaction.xsd", IsNullable=false)]
        public string[] transactionIdList {
            get {
                return this.transactionIdListField;
            }
            set {
                this.transactionIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("partyId", Namespace="http://cufxstandards.com/v3/Party.xsd", IsNullable=false)]
        public string[] partyIdList {
            get {
                return this.partyIdListField;
            }
            set {
                this.partyIdListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("accountId", Namespace="http://cufxstandards.com/v3/Account.xsd", IsNullable=false)]
        public string[] accountIdList {
            get {
                return this.accountIdListField;
            }
            set {
                this.accountIdListField = value;
            }
        }
        
        /// <remarks/>
        public TransactionType transactionType {
            get {
                return this.transactionTypeField;
            }
            set {
                this.transactionTypeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool transactionTypeSpecified {
            get {
                return this.transactionTypeFieldSpecified;
            }
            set {
                this.transactionTypeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("transactionStatus", Namespace="http://cufxstandards.com/v3/Transaction.xsd", IsNullable=false)]
        public TransactionStatus[] transactionStatusList {
            get {
                return this.transactionStatusListField;
            }
            set {
                this.transactionStatusListField = value;
            }
        }
        
        /// <remarks/>
        public Money transactionMinAmount {
            get {
                return this.transactionMinAmountField;
            }
            set {
                this.transactionMinAmountField = value;
            }
        }
        
        /// <remarks/>
        public Money transactionMaxAmount {
            get {
                return this.transactionMaxAmountField;
            }
            set {
                this.transactionMaxAmountField = value;
            }
        }
        
        /// <remarks/>
        public string descriptionContains {
            get {
                return this.descriptionContainsField;
            }
            set {
                this.descriptionContainsField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("checkNumber", Namespace="http://cufxstandards.com/v3/Transaction.xsd", IsNullable=false)]
        public string[] checkNumberList {
            get {
                return this.checkNumberListField;
            }
            set {
                this.checkNumberListField = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime transactionEffectiveStartDateTime {
            get {
                return this.transactionEffectiveStartDateTimeField;
            }
            set {
                this.transactionEffectiveStartDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool transactionEffectiveStartDateTimeSpecified {
            get {
                return this.transactionEffectiveStartDateTimeFieldSpecified;
            }
            set {
                this.transactionEffectiveStartDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime transactionEffectiveEndDateTime {
            get {
                return this.transactionEffectiveEndDateTimeField;
            }
            set {
                this.transactionEffectiveEndDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool transactionEffectiveEndDateTimeSpecified {
            get {
                return this.transactionEffectiveEndDateTimeFieldSpecified;
            }
            set {
                this.transactionEffectiveEndDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime transactionPostedStartDateTime {
            get {
                return this.transactionPostedStartDateTimeField;
            }
            set {
                this.transactionPostedStartDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool transactionPostedStartDateTimeSpecified {
            get {
                return this.transactionPostedStartDateTimeFieldSpecified;
            }
            set {
                this.transactionPostedStartDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        public System.DateTime transactionPostedEndDateTime {
            get {
                return this.transactionPostedEndDateTimeField;
            }
            set {
                this.transactionPostedEndDateTimeField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlIgnoreAttribute()]
        public bool transactionPostedEndDateTimeSpecified {
            get {
                return this.transactionPostedEndDateTimeFieldSpecified;
            }
            set {
                this.transactionPostedEndDateTimeFieldSpecified = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("category", Namespace="http://cufxstandards.com/v3/Transaction.xsd", IsNullable=false)]
        public string[] categoryList {
            get {
                return this.categoryListField;
            }
            set {
                this.categoryListField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("source", Namespace="http://cufxstandards.com/v3/Transaction.xsd", IsNullable=false)]
        public TransactionSource[] sourceList {
            get {
                return this.sourceListField;
            }
            set {
                this.sourceListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/TransactionMessage.xsd")]
    public partial class TransactionMessage {
        
        private MessageContext messageContextField;
        
        private TransactionFilter transactionFilterField;
        
        private TransactionListTransaction[] transactionListField;
        
        /// <remarks/>
        public MessageContext messageContext {
            get {
                return this.messageContextField;
            }
            set {
                this.messageContextField = value;
            }
        }
        
        /// <remarks/>
        public TransactionFilter transactionFilter {
            get {
                return this.transactionFilterField;
            }
            set {
                this.transactionFilterField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("transaction", Namespace="http://cufxstandards.com/v3/Transaction.xsd")]
        public TransactionListTransaction[] transactionList {
            get {
                return this.transactionListField;
            }
            set {
                this.transactionListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/TransferOccurrence.xsd")]
    public partial class TransferOccurrenceList {
        
        private TransferOccurrence[] transferOccurrenceField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("transferOccurrence")]
        public TransferOccurrence[] transferOccurrence {
            get {
                return this.transferOccurrenceField;
            }
            set {
                this.transferOccurrenceField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/TransferOccurrenceMessage.xsd")]
    public partial class TransferOccurrenceMessage {
        
        private MessageContext messageContextField;
        
        private TransferFilter transferFilterField;
        
        private TransferOccurrence[] transferOccurrenceListField;
        
        /// <remarks/>
        public MessageContext messageContext {
            get {
                return this.messageContextField;
            }
            set {
                this.messageContextField = value;
            }
        }
        
        /// <remarks/>
        public TransferFilter transferFilter {
            get {
                return this.transferFilterField;
            }
            set {
                this.transferFilterField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("transferOccurrence", Namespace="http://cufxstandards.com/v3/TransferOccurrence.xsd", IsNullable=false)]
        public TransferOccurrence[] transferOccurrenceList {
            get {
                return this.transferOccurrenceListField;
            }
            set {
                this.transferOccurrenceListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/TransferRecurring.xsd")]
    public partial class TransferRecurringList {
        
        private TransferRecurring[] transferRecurringField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("transferRecurring")]
        public TransferRecurring[] transferRecurring {
            get {
                return this.transferRecurringField;
            }
            set {
                this.transferRecurringField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/TransferRecurringMessage.xsd")]
    public partial class TransferRecurringMessage {
        
        private MessageContext messageContextField;
        
        private TransferFilter transferFilterField;
        
        private TransferRecurring[] transferRecurringListField;
        
        /// <remarks/>
        public MessageContext messageContext {
            get {
                return this.messageContextField;
            }
            set {
                this.messageContextField = value;
            }
        }
        
        /// <remarks/>
        public TransferFilter transferFilter {
            get {
                return this.transferFilterField;
            }
            set {
                this.transferFilterField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("transferRecurring", Namespace="http://cufxstandards.com/v3/TransferRecurring.xsd", IsNullable=false)]
        public TransferRecurring[] transferRecurringList {
            get {
                return this.transferRecurringListField;
            }
            set {
                this.transferRecurringListField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/ValidationStatus.xsd")]
    public enum ValidationStatus {
        
        /// <remarks/>
        Pass,
        
        /// <remarks/>
        Fail,
        
        /// <remarks/>
        Indeterminate,
        
        /// <remarks/>
        Error,
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/Wire.xsd")]
    public partial class WireList {
        
        private Wire[] wireField;
        
        /// <remarks/>
        [System.Xml.Serialization.XmlElementAttribute("wire")]
        public Wire[] wire {
            get {
                return this.wireField;
            }
            set {
                this.wireField = value;
            }
        }
    }
    
    /// <remarks/>
    [System.Xml.Serialization.XmlTypeAttribute(Namespace="http://cufxstandards.com/v3/WireMessage.xsd")]
    public partial class WireMessage {
        
        private MessageContext messageContextField;
        
        private WireFilter wireFilterField;
        
        private Wire[] wireListField;
        
        /// <remarks/>
        public MessageContext messageContext {
            get {
                return this.messageContextField;
            }
            set {
                this.messageContextField = value;
            }
        }
        
        /// <remarks/>
        public WireFilter wireFilter {
            get {
                return this.wireFilterField;
            }
            set {
                this.wireFilterField = value;
            }
        }
        
        /// <remarks/>
        [System.Xml.Serialization.XmlArrayItemAttribute("wire", Namespace="http://cufxstandards.com/v3/Wire.xsd", IsNullable=false)]
        public Wire[] wireList {
            get {
                return this.wireListField;
            }
            set {
                this.wireListField = value;
            }
        }
    }
}
