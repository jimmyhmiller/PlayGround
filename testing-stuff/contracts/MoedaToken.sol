import './StandardToken.sol';
import './Ownable.sol';

pragma solidity ^0.4.11;

/// @title Moeda Loaylty Points token contract
contract MoedaToken is StandardToken, Ownable {
    string public constant name = "Moeda Loyalty Points";
    string public constant symbol = "MDA";
    uint8 public constant decimals = 18;

    // don't allow creation of more than this number of tokens
    uint public constant MAX_TOKENS = 20000000 ether;
    
    // transfers are locked during the sale
    bool public saleActive;

    // only emitted during the crowdsale
    event Created(address indexed donor, uint256 tokensReceived);

    // determine whether transfers can be made
    modifier onlyAfterSale() {
        if (saleActive) {
            throw;
        }
        _;
    }

    modifier onlyDuringSale() {
        if (!saleActive) {
            throw;
        }
        _;
    }

    /// @dev Create moeda token and lock transfers
    function MoedaToken() {
        saleActive = true;
    }

    /// @dev unlock transfers
    function unlock() onlyOwner {
        saleActive = false;
    }

    /// @dev create tokens, only usable while saleActive
    /// @param recipient address that will receive the created tokens
    /// @param amount the number of tokens to create
    function create(address recipient, uint256 amount)
    onlyOwner onlyDuringSale {
        if (amount == 0) throw;
        if (safeAdd(totalSupply, amount) > MAX_TOKENS) throw;

        balances[recipient] = safeAdd(balances[recipient], amount);
        totalSupply = safeAdd(totalSupply, amount);

        Created(recipient, amount);
    }

    // transfer tokens
    // only allowed after sale has ended
    function transfer(address _to, uint _value) onlyAfterSale returns (bool) {
        return super.transfer(_to, _value);
    }

    // transfer tokens
    // only allowed after sale has ended
    function transferFrom(address from, address to, uint value) onlyAfterSale 
    returns (bool)
    {
        return super.transferFrom(from, to, value);
    }
}