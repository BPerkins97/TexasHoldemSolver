package icybee.solver.nodes;

/**
 * Created by huangxuefeng on 2019/10/7.
 * This file contains code for game actions.
 */
public class GameActions {
    GameTreeNode.PokerActions action;
    Double amount;

    public GameActions(GameTreeNode.PokerActions action, Double amount) {
        this.action = action;
        if (action == GameTreeNode.PokerActions.RAISE || action == GameTreeNode.PokerActions.BET) {
            assert (amount != null);
        } else {
            assert (amount == null);
        }
        this.amount = amount;
    }

    public GameTreeNode.PokerActions getAction() {
        return action;
    }

    public double getAmount() {
        return amount;
    }

    @Override
    public String toString() {
        if (amount == null) {
            return this.action.toString();
        } else {
            return this.action.toString() + " " + amount.toString();
        }
    }
}
