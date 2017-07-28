import './SafeMath.sol';
import './Ownable.sol';
import './MoedaToken.sol';
import './ERC20.sol';

pragma solidity ^0.4.11;

/// @title Moeda crowdsale
contract Crowdsale is Ownable, SafeMath {
    bool public crowdsaleClosed;        // whether the crowdsale has been closed 
                                        // manually
    address public wallet;              // recipient of all crowdsale funds
    MoedaToken public moedaToken;       // token that will be sold during sale
    uint256 public etherReceived;       // total ether received
    uint256 public totalTokensSold;     // number of tokens sold
    uint256 public startBlock;          // block where sale starts
    uint256 public endBlock;            // block where sale ends

    // used to scale token amounts to 18 decimals
    uint256 public constant TOKEN_MULTIPLIER = 10 ** 18;

    // number of tokens allocated to presale (prior to crowdsale)
    uint256 public constant PRESALE_TOKEN_ALLOCATION = 5000000 * TOKEN_MULTIPLIER;

    // recipient of presale tokens
    address public PRESALE_WALLET = "0x30B3C64d43e7A1E8965D934Fa96a3bFB33Eee0d2";
    
    // smallest possible donation
    uint256 public constant DUST_LIMIT = 1 finney;

    // token generation rates (tokens per eth)
    uint256 public constant TIER1_RATE = 160;
    uint256 public constant TIER2_RATE = 125;
    uint256 public constant TIER3_RATE = 80;

    // limits for each pricing tier (how much can be bought)
    uint256 public constant TIER1_CAP =  31250 ether;
    uint256 public constant TIER2_CAP =  71250 ether;
    uint256 public constant TIER3_CAP = 133750 ether; // Total ether cap

    // Log a purchase
    event Purchase(address indexed donor, uint256 amount, uint256 tokenAmount);

    // Log transfer of tokens that were sent to this contract by mistake
    event TokenDrain(address token, address to, uint256 amount);

    modifier onlyDuringSale() {
        if (crowdsaleClosed) {
            throw;
        }

        if (block.number < startBlock) {
            throw;
        }

        if (block.number >= endBlock) {
            throw;
        }
        _;
    }

    /// @dev Initialize a new Crowdsale contract
    /// @param _wallet address of multisig wallet that will store received ether
    /// @param _startBlock block at which to start the sale
    /// @param _endBlock block at which to end the sale
    function Crowdsale(address _wallet, uint _startBlock, uint _endBlock) {
        if (_wallet == address(0)) throw;
        if (_startBlock <= block.number) throw;
        if (_endBlock <= _startBlock) throw;
        
        crowdsaleClosed = false;
        wallet = _wallet;
        moedaToken = new MoedaToken();
        startBlock = _startBlock;
        endBlock = _endBlock;
    }

    /// @dev Determine the lowest rate to acquire tokens given an amount of 
    /// donated ethers
    /// @param totalReceived amount of ether that has been received
    /// @return pair of the current tier's donation limit and a token creation rate
    function getLimitAndPrice(uint256 totalReceived)
    constant returns (uint256, uint256) {
        uint256 limit = 0;
        uint256 price = 0;

        if (totalReceived < TIER1_CAP) {
            limit = TIER1_CAP;
            price = TIER1_RATE;
        }
        else if (totalReceived < TIER2_CAP) {
            limit = TIER2_CAP;
            price = TIER2_RATE;
        }
        else if (totalReceived < TIER3_CAP) {
            limit = TIER3_CAP;
            price = TIER3_RATE;
        } else {
            throw; // this shouldn't happen
        }

        return (limit, price);
    }

    /// @dev Determine how many tokens we can get from each pricing tier, in
    /// case a donation's amount overlaps multiple pricing tiers.
    ///
    /// @param totalReceived ether received by contract plus spent by this donation
    /// @param requestedAmount total ether to spend on tokens in a donation
    /// @return amount of tokens to get for the requested ether donation
    function getTokenAmount(uint256 totalReceived, uint256 requestedAmount) 
    constant returns (uint256) {

        // base case, we've spent the entire donation and can stop
        if (requestedAmount == 0) return 0;
        uint256 limit = 0;
        uint256 price = 0;
        
        // 1. Determine cheapest token price
        (limit, price) = getLimitAndPrice(totalReceived);

        // 2. Since there are multiple pricing levels based on how much has been
        // received so far, we need to determine how much can be spent at
        // any given tier. This in case a donation will overlap more than one 
        // tier
        uint256 maxETHSpendableInTier = safeSub(limit, totalReceived);
        uint256 amountToSpend = min256(maxETHSpendableInTier, requestedAmount);

        // 3. Given a price determine how many tokens the unspent ether in this 
        // donation will get you
        uint256 tokensToReceiveAtCurrentPrice = safeMul(amountToSpend, price);

        // You've spent everything you could at this level, continue to the next
        // one, in case there is some ETH left unspent in this donation.
        uint256 additionalTokens = getTokenAmount(
            safeAdd(totalReceived, amountToSpend),
            safeSub(requestedAmount, amountToSpend));

        return safeAdd(tokensToReceiveAtCurrentPrice, additionalTokens);
    }

    /// grant tokens to buyer when we receive ether
    /// @dev buy tokens, only usable while crowdsale is active
    function () payable onlyDuringSale {
        if (msg.value < DUST_LIMIT) throw;
        if (safeAdd(etherReceived, msg.value) > TIER3_CAP) throw;

        uint256 tokenAmount = getTokenAmount(etherReceived, msg.value);

        moedaToken.create(msg.sender, tokenAmount);
        etherReceived = safeAdd(etherReceived, msg.value);
        totalTokensSold = safeAdd(totalTokensSold, tokenAmount);
        Purchase(msg.sender, msg.value, tokenAmount);

        if (!wallet.send(msg.value)) throw;
    }

    /// @dev close the crowdsale manually and unlock the tokens
    /// this will only be successful if not already executed,
    /// if endBlock has been reached, or if the cap has been reached
    function finalize() onlyOwner {
        if (block.number < startBlock) throw;
        if (crowdsaleClosed) throw;

        // if amount remaining is too small we can allow sale to end earlier
        uint256 amountRemaining = safeSub(TIER3_CAP, etherReceived);
        if (block.number < endBlock && amountRemaining >= DUST_LIMIT) throw;

        // create and assign presale tokens to presale wallet
        moedaToken.create(PRESALE_WALLET, PRESALE_TOKEN_ALLOCATION);

        // unlock tokens for spending
        moedaToken.unlock();
        crowdsaleClosed = true;
    }

    /// @dev Drain tokens that were sent here by mistake
    /// because people will.
    /// @param _token address of token to transfer
    /// @param _to address where tokens will be transferred
    function drainToken(address _token, address _to) onlyOwner {
        if (_token == address(0)) throw;
        if (_to == address(0)) throw;
        ERC20 token = ERC20(_token);
        uint256 balance = token.balanceOf(this);
        token.transfer(_to, balance);
        TokenDrain(_token, _to, balance);
    }
}
